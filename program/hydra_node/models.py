import os
import bisect
import json
from re import S
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from dotmap import DotMap
import numpy as np
from torch import autocast
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.attention import CrossAttention, HyperLogic
from PIL import Image
import k_diffusion as K
import contextlib
import random

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pil_upscale(image, scale=1):
    device = image.device
    dtype = image.dtype
    image = Image.fromarray((image.cpu().permute(1,2,0).numpy().astype(np.float32) * 255.).astype(np.uint8))
    if scale > 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), resample=Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.
    image = repeat(image, '1 ... -> b ...', b=1)
    return image.to(device)

def fix_batch(tensor, bs):
    return torch.stack([tensor.squeeze(0)]*bs, dim=0)

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# make uc and prompt shapes match via padding for long prompts
# finetune
null_cond = None
def fix_cond_shapes(model, prompt_condition, uc):
    global null_cond
    if null_cond is None:
        null_cond = model.get_learned_conditioning([""])
    while prompt_condition.shape[1] > uc.shape[1]:
        uc = torch.cat((uc, null_cond.repeat((uc.shape[0], 1, 1))), axis=1)
    while prompt_condition.shape[1] < uc.shape[1]:
        prompt_condition = torch.cat((prompt_condition, null_cond.repeat((prompt_condition.shape[0], 1, 1))), axis=1)
    return prompt_condition, uc

# mix conditioning vectors for prompts
# @aero
def prompt_mixing(model, prompt_body, batch_size):
    if "|" in prompt_body:
        prompt_parts = prompt_body.split("|")
        prompt_total_power = 0
        prompt_sum = None
        for prompt_part in prompt_parts:
            prompt_power = 1
            if ":" in prompt_part:
                prompt_sub_parts = prompt_part.split(":")
                try:
                    prompt_power = float(prompt_sub_parts[1])
                    prompt_part = prompt_sub_parts[0]
                except:
                    print("Error parsing prompt power! Assuming 1")
            prompt_vector = model.get_learned_conditioning([prompt_part])
            if prompt_sum is None:
                prompt_sum = prompt_vector * prompt_power
            else:
                prompt_sum, prompt_vector = fix_cond_shapes(model, prompt_sum, prompt_vector)
                prompt_sum = prompt_sum + (prompt_vector * prompt_power)
            prompt_total_power = prompt_total_power + prompt_power
        return fix_batch(prompt_sum / prompt_total_power, batch_size)
    else:
        return fix_batch(model.get_learned_conditioning([prompt_body]), batch_size)

def sample_start_noise(seed, C, H, W, f, device="cuda"):
    if seed:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn([C, (H) // f, (W) // f], generator=gen, device=device).unsqueeze(0)
    else:
        noise = torch.randn([C, (H) // f, (W) // f], device=device).unsqueeze(0)
    return noise

def sample_start_noise_special(seed, request, device="cuda"):
    if seed:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn([request.latent_channels, request.height // request.downsampling_factor, request.width // request.downsampling_factor], generator=gen, device=device).unsqueeze(0)
    else:
        noise = torch.randn([request.latent_channels, request.height // request.downsampling_factor, request.width // request.downsampling_factor], device=device).unsqueeze(0)
    return noise

@torch.no_grad()
#@torch.autocast("cuda", enabled=True, dtype=torch.float16)
def encode_image(image, model):
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = torch.from_numpy(image)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    #dtype = image.dtype
    image = image.to(torch.float32)
    #gets image as numpy array and returns as tensor
    def preprocess_vqgan(x):
        x = x / 255.0
        x = 2.*x - 1.
        return x

    image = image.permute(2, 0, 1).unsqueeze(0).float().cuda()
    image = preprocess_vqgan(image)
    image = model.encode(image).sample()
    #image = image.to(dtype)

    return image

@torch.no_grad()
def decode_image(image, model):
    def custom_to_pil(x):
        x = x.detach().float().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        x = x.permute(0, 2, 3, 1)#.numpy()
        #x = (255*x).astype(np.uint8)
        #x = Image.fromarray(x)
        #if not x.mode == "RGB":
        #    x = x.convert("RGB")
        return x

    image = model.decode(image)
    image = custom_to_pil(image)
    return image

class VectorAdjustPrior(nn.Module):
    def __init__(self, hidden_size, inter_dim=64):
        super().__init__()
        self.vector_proj = nn.Linear(hidden_size*2, inter_dim, bias=True)
        self.out_proj = nn.Linear(hidden_size+inter_dim, hidden_size, bias=True)

    def forward(self, z):
        b, s = z.shape[0:2]
        x1 = torch.mean(z, dim=1).repeat(s, 1)
        x2 = z.reshape(b*s, -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.vector_proj(x)
        x = torch.cat((x2, x), dim=1)
        x = self.out_proj(x)
        x = x.reshape(b, s, -1)
        return x

    @classmethod
    def load_model(cls, model_path, hidden_size=768, inter_dim=64):
        model = cls(hidden_size=hidden_size, inter_dim=inter_dim)
        model.load_state_dict(torch.load(model_path)["state_dict"])
        return model

class StableInterface(nn.Module):
    def __init__(self, model, thresholder = None):
        super().__init__()
        self.inner_model = model
        self.sigma_to_t = model.sigma_to_t
        self.thresholder = thresholder
        self.get_sigmas = model.get_sigmas

    @torch.no_grad()
    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_two = torch.cat([x] * 2)
        sigma_two = torch.cat([sigma] * 2)
        cond_full = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_two, sigma_two, cond=cond_full).chunk(2)
        x_0 = uncond + (cond - uncond) * cond_scale
        if self.thresholder is not None:
            x_0 = self.thresholder(x_0)

        return x_0

class StableDiffusionModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.premodules = None
        if Path(self.config.model_path).is_dir():
            config.logger.info(f"Loading model from folder {self.config.model_path}")
            model, model_config = self.from_folder(config.model_path)

        elif Path(self.config.model_path).is_file():
            config.logger.info(f"Loading model from file {self.config.model_path}")
            model, model_config = self.from_file(config.model_path)

        else:
            raise Exception("Invalid model path!")

        if config.dtype == "float16":
            typex = torch.float16
        else:
            typex = torch.float32
        self.model = model.to(config.device).to(typex)
        if self.config.vae_path:
            ckpt=torch.load(self.config.vae_path, map_location="cpu")
            loss = []
            for i in ckpt["state_dict"].keys():
                if i[0:4] == "loss":
                    loss.append(i)
            for i in loss:
                del ckpt["state_dict"][i]

            model.first_stage_model = model.first_stage_model.float()
            model.first_stage_model.load_state_dict(ckpt["state_dict"])
            model.first_stage_model = model.first_stage_model.float()
            del ckpt
            del loss
            config.logger.info(f"Using VAE from {self.config.vae_path}")

        if self.config.penultimate == "1":
            model.cond_stage_model.return_layer = -2
            model.cond_stage_model.do_final_ln = True
            config.logger.info(f"CLIP: Using penultimate layer")

        if self.config.clip_contexts > 1:
            model.cond_stage_model.clip_extend = True
            model.cond_stage_model.max_clip_extend = 75 * self.config.clip_contexts

        model.cond_stage_model.inference_mode = True
        self.k_model = K.external.CompVisDenoiser(model)
        self.k_model = StableInterface(self.k_model)
        self.device = config.device
        self.model_config = model_config
        self.plms = PLMSSampler(model)
        self.ddim = DDIMSampler(model)
        self.ema_manager = self.model.ema_scope
        if self.config.enable_ema == "0":
            self.ema_manager = contextlib.nullcontext
            config.logger.info("Disabling EMA")
        else:
            config.logger.info(f"Using EMA")
        self.sampler_map = {
            'plms': self.plms.sample,
            'ddim': self.ddim.sample,
            'k_euler': K.sampling.sample_euler,
            'k_euler_ancestral': K.sampling.sample_euler_ancestral,
            'k_heun': K.sampling.sample_heun,
            'k_dpm_2': K.sampling.sample_dpm_2,
            'k_dpm_2_ancestral': K.sampling.sample_dpm_2_ancestral,
            'k_lms': K.sampling.sample_lms,
        }
        if config.prior_path:
            self.prior = VectorAdjustPrior.load_model(config.prior_path).to(self.device)
        self.copied_ema = False

    @property
    def get_default_config(self):
        dict_config = {
            'steps': 30,
            'sampler': "k_euler_ancestral",
            'n_samples': 1,
            'image': None,
            'fixed_code': False,
            'ddim_eta': 0.0,
            'height': 512,
            'width': 512,
            'latent_channels': 4,
            'downsampling_factor': 8,
            'scale': 12.0,
            'dynamic_threshold': None,
            'seed': None,
            'stage_two_seed': None,
            'module': None,
            'masks': None,
            'output': None,
        }
        return DotMap(dict_config)

    def from_folder(self, folder):
        folder = Path(folder)
        model_config = OmegaConf.load(folder / "config.yaml")
        if (folder / "pruned.ckpt").is_file():
            model_path = folder / "pruned.ckpt"
        else:
            model_path = folder / "model.ckpt"
        model = self.load_model_from_config(model_config, model_path)
        return model, model_config

    def from_path(self, file):
        default_config = Path(self.config.default_config)
        if not default_config.is_file():
            raise Exception("Default config to load not found! Either give a folder on MODEL_PATH or specify a config to use with this checkpoint on DEFAULT_CONFIG")
        model_config = OmegaConf.load(default_config)
        model = self.load_model_from_config(model_config, file)
        return model, model_config

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")

        sd = pl_sd.get('state_dict', pl_sd)

        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.eval()
        return model

    @torch.no_grad()
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def sample(self, request):
        if request.module is not None:
            if request.module == "vanilla":
                pass

            else:
                module = self.premodules[request.module]
                CrossAttention.set_hypernetwork(module)

        if request.seed is not None:
            seed_everything(request.seed)

        if request.image is not None:
            request.steps = 50
            #request.sampler = "ddim_img2img" #enforce ddim for now
            if request.sampler == "plms":
                request.sampler = "k_lms"
            if request.sampler == "ddim":
                request.sampler = "k_lms"

            self.ddim.make_schedule(ddim_num_steps=request.steps, ddim_eta=request.ddim_eta, verbose=False)
            start_code = encode_image(request.image, self.model.first_stage_model).to(self.device)
            start_code = self.model.get_first_stage_encoding(start_code)
            start_code = torch.repeat_interleave(start_code, request.n_samples, dim=0)

            main_noise = []
            start_noise = []
            for seed in range(request.seed, request.seed+request.n_samples):
                main_noise.append(sample_start_noise(seed, request.latent_channels, request.height, request.width, request.downsampling_factor, self.device))
                start_noise.append(sample_start_noise(seed, request.latent_channels, request.height, request.width, request.downsampling_factor, self.device))

            main_noise = torch.cat(main_noise, dim=0)
            start_noise = torch.cat(start_noise, dim=0)

            start_code = start_code + (start_noise * request.noise)
            t_enc = int(request.strength * request.steps)

        if request.sampler.startswith("k_"):
            sampler = "k-diffusion"

        elif request.sampler == 'ddim_img2img':
            sampler = 'img2img'

        else:
            sampler = "normal"

        if request.image is None:
            main_noise = []
            for seed_offset in range(request.n_samples):
                if request.masks is not None:
                    noise_x = sample_start_noise_special(request.seed, request, self.device)
                else:
                    noise_x = sample_start_noise_special(request.seed+seed_offset, request, self.device)

                if request.masks is not None:
                    for maskobj in request.masks:
                        mask_seed = maskobj["seed"]
                        mask = maskobj["mask"]
                        mask = np.asarray(mask)
                        mask = torch.from_numpy(mask).clone().to(self.device).permute(2, 0, 1)
                        mask = mask.float() / 255.0
                        # convert RGB or grayscale image into 4-channel
                        mask = mask[0].unsqueeze(0)
                        mask = torch.repeat_interleave(mask, request.latent_channels, dim=0)
                        mask = (mask < 0.5).float()

                        # interpolate start noise
                        noise_x = (noise_x * (1-mask)) + (sample_start_noise_special(mask_seed+seed_offset, request, self.device) * mask)

                main_noise.append(noise_x)

            main_noise = torch.cat(main_noise, dim=0)
            start_code = main_noise

        prompt = [request.prompt]
        prompt_condition = prompt_mixing(self.model, prompt[0], 1)
        if hasattr(self, "prior") and request.mitigate:
            prompt_condition = self.prior(prompt_condition)

        uc = None
        if request.scale != 1.0:
            if request.uc is not None:
                uc = [request.uc]
                uc = prompt_mixing(self.model, uc[0], 1)
            else:
                if self.config.quality_hack == "1":
                    uc = ["Tags: lowres"]
                    uc = prompt_mixing(self.model, uc[0], 1)
                else:
                    uc = self.model.get_learned_conditioning([""])
            prompt_condition, uc = fix_cond_shapes(self.model, prompt_condition, uc)

        shape = [
            request.latent_channels,
            request.height // request.downsampling_factor,
            request.width // request.downsampling_factor
        ]

        # handle images one at a time because batches eat absurd VRAM
        sampless = []
        for main_noise, start_code in zip(main_noise.chunk(request.n_samples), start_code.chunk(request.n_samples)):
            if sampler == "normal":
                with self.ema_manager():
                    samples, _ = self.sampler_map[request.sampler](
                        S=request.steps,
                        conditioning=prompt_condition,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=request.scale,
                        unconditional_conditioning=uc,
                        eta=request.ddim_eta,
                        dynamic_threshold=request.dynamic_threshold,
                        x_T=start_code,
                    )

            elif sampler == "k-diffusion":
                with self.ema_manager():
                    sigmas = self.k_model.get_sigmas(request.steps)
                    if request.image is not None:
                        noise = main_noise * sigmas[request.steps - t_enc - 1]
                        start_code = start_code + noise
                        sigmas = sigmas[request.steps - t_enc - 1:]

                    else:
                        start_code = start_code * sigmas[0]

                    extra_args = {'cond': prompt_condition, 'uncond': uc, 'cond_scale': request.scale}
                    samples = self.sampler_map[request.sampler](self.k_model, start_code, sigmas, request.seed, extra_args=extra_args)

            sampless.append(samples)
            torch_gc()

        images = []
        for samples in sampless:
            with torch.autocast("cuda", enabled=False):
                x_samples_ddim = self.model.decode_first_stage(samples.float())
                #x_samples_ddim = decode_image(samples, self.model.first_stage_model)
                #x_samples_ddim = self.model.first_stage_model.decode(samples.float())
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                x_sample = np.ascontiguousarray(x_sample)
                images.append(x_sample)

            torch_gc()

        if request.seed is not None:
            torch.seed()
            np.random.seed()

        #set hypernetwork to none after generation
        CrossAttention.set_hypernetwork(None)

        return images

    @torch.no_grad()
    def sample_two_stages(self, request):
        request = DotMap(request)
        if request.seed is not None:
            seed_everything(request.seed)

        if request.plms:
            sampler = self.plms
        else:
            sampler = self.ddim

        start_code = None
        if request.fixed_code:
            start_code = torch.randn([
                request.n_samples,
                request.latent_channels,
                request.height // request.downsampling_factor,
                request.width // request.downsampling_factor,
                ], device=self.device)

        prompt = [request.prompt] * request.n_samples
        prompt_condition = prompt_mixing(self.model, prompt[0], request.n_samples)

        uc = None
        if request.scale != 1.0:
            uc = self.model.get_learned_conditioning(request.n_samples * [""])
            prompt_condition, uc = fix_cond_shapes(self.model, prompt_condition, uc)

        shape = [
            request.latent_channels,
            request.height // request.downsampling_factor,
            request.width // request.downsampling_factor
        ]
        with torch.autocast("cuda", enabled=self.config.amp):
            with ema_manager():
                samples, _ = sampler.sample(
                    S=request.steps,
                    conditioning=prompt_condition,
                    batch_size=request.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=request.scale,
                    unconditional_conditioning=uc,
                    eta=request.ddim_eta,
                    dynamic_threshold=request.dynamic_threshold,
                    x_T=start_code,
                )

        x_samples_ddim = self.model.decode_first_stage(samples)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).squeeze(0)
        x_samples_ddim = pil_upscale(x_samples_ddim, scale=2)

        if request.stage_two_seed is not None:
            torch.manual_seed(request.stage_two_seed)
            np.random.seed(request.stage_two_seed)

        with torch.autocast("cuda", enabled=self.config.amp):
            with ema_manager():
                init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(x_samples_ddim))
                self.ddim.make_schedule(ddim_num_steps=request.steps, ddim_eta=request.ddim_eta, verbose=False)
                t_enc = int(request.strength * request.steps)

                print("init latent shape:")
                print(init_latent.shape)

                init_latent = init_latent + (torch.randn_like(init_latent) * request.noise)

                prompt_condition = prompt_mixing(self.model, prompt[0], request.n_samples)

                uc = None
                if request.scale != 1.0:
                    uc = self.model.get_learned_conditioning(request.n_samples * [""])
                    prompt_condition, uc = fix_cond_shapes(self.model, prompt_condition, uc)

                # encode (scaled latent)
                start_code_terped=None
                z_enc = self.ddim.stochastic_encode(init_latent, torch.tensor([t_enc]*request.n_samples).to(self.device), noise=start_code_terped)
                # decode it
                samples = self.ddim.decode(z_enc, prompt_condition, t_enc, unconditional_guidance_scale=request.scale,
                                        unconditional_conditioning=uc,)

                x_samples_ddim = self.model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        images = []
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            x_sample = x_sample.astype(np.uint8)
            x_sample = np.ascontiguousarray(x_sample)
            images.append(x_sample)

        if request.seed is not None:
            torch.seed()
            np.random.seed()

        return images

    @torch.no_grad()
    def sample_from_image(self, request):
        return

class DalleMiniModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        from min_dalle import MinDalle

        self.config = config
        self.model = MinDalle(
            models_root=config.model_path,
            dtype=torch.float16,
            device='cuda',
            is_mega=True,
            is_reusable=True
        )

    @torch.no_grad()
    def sample(self, request):
        if request.seed is not None:
            seed = request.seed
        else:
            seed = -1

        images = self.model.generate_images(
            text=request.prompt,
            seed=seed,
            grid_size=request.grid_size,
            is_seamless=False,
            temperature=request.temp,
            top_k=request.top_k,
            supercondition_factor=request.scale,
            is_verbose=False
        )
        images = images.to('cpu').numpy()
        images = images.astype(np.uint8)
        images = np.ascontiguousarray(images)

        if request.seed is not None:
            torch.seed()
            np.random.seed()

        return images

def apply_temp(logits, temperature):
    logits = logits / temperature
    return logits

@torch.no_grad()
def generate(forward, prompt_tokens, tokenizer, tokens_to_generate=50, ds=False, ops_list=[{"temp": 0.9}], hypernetwork=None, non_deterministic=False, fully_deterministic=False):
    in_tokens = prompt_tokens
    context = prompt_tokens
    generated = torch.zeros(len(ops_list), 0, dtype=torch.long).to(in_tokens.device)
    kv = None
    if non_deterministic:
        torch.seed()
    #soft_required = ["top_k", "top_p"]
    op_map = {
        "temp": apply_temp,
    }

    for _ in range(tokens_to_generate):
        if ds:
            logits, kv = forward(in_tokens, past_key_values=kv, use_cache=True)
        else:
            logits, kv = forward(in_tokens, cache=True, kv=kv, hypernetwork=hypernetwork)
        logits = logits[:, -1, :] #get the last token in the seq
        logits = torch.log_softmax(logits, dim=-1)

        batch = []
        for i, ops in enumerate(ops_list):
            item = logits[i, ...].unsqueeze(0)
            ctx = context[i, ...].unsqueeze(0)
            for op, value in ops.items():
                if op == "rep_pen":
                    item = op_map[op](ctx, item, **value)

                else:
                    item = op_map[op](item, value)

            batch.append(item)

        logits = torch.cat(batch, dim=0)
        logits = torch.softmax(logits, dim=-1)

        #fully_deterministic makes it deterministic across the batch
        if fully_deterministic:
            logits = logits.split(1, dim=0)
            logit_list = []
            for logit in logits:
                torch.manual_seed(69)
                logit_list.append(torch.multinomial(logit, 1))

            logits = torch.cat(logit_list, dim=0)

        else:
            logits = torch.multinomial(logits, 1)

        if logits[0, 0] == 48585:
            if generated[0, -1] == 1400:
                pass
            elif generated[0, -1] == 3363:
                return "safe", "none"
            else:
                return "notsafe", tokenizer.decode(generated.squeeze()).split("Output: ")[-1]

        generated = torch.cat([generated, logits], dim=-1)
        context = torch.cat([context, logits], dim=-1)
        in_tokens = logits

    return "unknown", tokenizer.decode(generated.squeeze())


class BasedformerModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        from basedformer import lm_utils
        from transformers import GPT2TokenizerFast
        self.config = config
        self.model = lm_utils.load_from_path(config.model_path).half().cuda()
        self.model = self.model.convert_to_ds()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    @torch.no_grad()
    def sample(self, request):
        prompt = request.prompt
        prompt = self.tokenizer.encode("Input: " + prompt, return_tensors='pt').cuda().long()
        prompt = torch.cat([prompt, torch.tensor([[49527]], dtype=torch.long).cuda()], dim=1)
        is_safe, corrected = generate(self.model.module, prompt, self.tokenizer, tokens_to_generate=150, ds=True)
        return is_safe, corrected

class EmbedderModel(nn.Module):
    def __init__(self, config=None):
        nn.Module.__init__(self)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('./models/sentence-transformers_all-MiniLM-L6-v2').to("cpu")
        self.knn = self.load_knn("models/tags.index")
        print("Loaded tag suggestion model using phrase embeddings")

    def load_knn(self, filename):
        import faiss
        try:
            return faiss.read_index(filename)
        except RuntimeError:
            print(f"Generating tag embedding index for {len(self.tags)} tags.")
            i = faiss.IndexFlatL2(384)
            i.add(self([name for name, count in self.tags]))
            faiss.write_index(i, filename)
            return i

    def __call__(self, sentences):
        with torch.no_grad():
            sentence_embeddings = self.model.encode(sentences)
            return sentence_embeddings

    def get_top_k(self, text):
        #check if text is a substring in tag_count.keys()
        found = []
        a = bisect.bisect_left(self.tags, (text,))
        b = bisect.bisect_left(self.tags, (text + '\xff',), lo=a)
        for tag, count in self.tags[a:b]:
            if len(tag) >= len(text) and tag.startswith(text):
                found.append([tag, count, 0])

        results = []
        embedding = self([text])
        k = 15
        D, I = self.knn.search(embedding, k)
        D, I = D.squeeze(), I.squeeze()
        for id, prob in zip(I, D):
            tag, count = self.tags[id]
            results.append([tag, count, prob])

        found.sort(key=lambda x: x[1], reverse=True)
        found = found[:5]
        # found = heapq.nlargest(5, found, key=lambda x: x[1])
        results_tags = [x[0] for x in found]
        for result in results.copy():
            if result[0] in results_tags:
                results.remove(result)

        results = sorted(results, key=lambda x: x[2], reverse=True)
        #filter results for >0.5 confidence unless it has the search text in it and confidence is >0.4
        results = [x for x in results if x[2] > 0.5 or (x[2] > 0.4 and text in x[0])]
        if found:
            results = found + results

        #max 10 results
        results = results[:10]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
