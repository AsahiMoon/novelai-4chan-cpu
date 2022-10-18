import traceback
from dotmap import DotMap
import math
from io import BytesIO
import base64
import random

v1pp_defaults = {
    'steps': 50,
    'sampler': "plms",
    'image': None,
    'fixed_code': False,
    'ddim_eta': 0.0,
    'height': 512,
    'width': 512,
    'latent_channels': 4,
    'downsampling_factor': 8,
    'scale': 7.0,
    'dynamic_threshold': None,
    'seed': None,
    'stage_two_seed': None,
    'module': None,
    'masks': None,
}

v1pp_forced_defaults = {
    'latent_channels': 4,
    'downsampling_factor': 8,
}

dalle_mini_defaults = {
    'temp': 1.0,
    'top_k': 256,
    'scale': 16,
    'grid_size': 4,
}

dalle_mini_forced_defaults = {
}

defaults = {
    'stable-diffusion': (v1pp_defaults, v1pp_forced_defaults),
    'dalle-mini': (dalle_mini_defaults, dalle_mini_forced_defaults),
    'basedformer': ({}, {}),
    'embedder': ({}, {}),
}

samplers = [
    "plms",
    "ddim",
    "k_euler",
    "k_euler_ancestral",
    "k_heun",
    "k_dpm_2",
    "k_dpm_2_ancestral",
    "k_lms"
    ]

def closest_multiple(num, mult):
    num_int = int(num)
    floor = math.floor(num_int / mult) * mult
    ceil = math.ceil(num_int / mult) * mult
    return floor if (num_int - floor) < (ceil - num_int) else ceil

def sanitize_stable_diffusion(request, config):
    if request.steps > 50:
        return False, "steps must be smaller than 50"

    if request.width * request.height == 0:
        return False, "width and height must be non-zero"

    if request.width <= 0:
        return False, "width must be positive"

    if request.height <= 0:
        return False, "height must be positive"

    if request.steps <= 0:
        return False, "steps must be positive"

    if request.ddim_eta < 0:
        return False, "ddim_eta shouldn't be negative"

    if request.scale < 1.0:
        return False, "scale should be at least 1.0"

    if request.dynamic_threshold is not None and request.dynamic_threshold < 0:
        return False, "dynamic_threshold shouldn't be negative"

    if request.width * request.height >= 1024*1025:
        return False, "width and height must be less than 1024*1025"

    if request.strength < 0.0 or request.strength >= 1.0:
        return False, "strength should be more than 0.0 and less than 1.0"

    if request.noise < 0.0 or request.noise > 1.0:
        return False, "noise should be more than 0.0 and less than 1.0"

    if request.advanced:
        request.width = closest_multiple(request.width // 2, 64)
        request.height = closest_multiple(request.height // 2, 64)

    if request.sampler not in samplers:
        return False, "sampler should be one of {}".format(samplers)

    if request.seed is None:
        state = random.getstate()
        request.seed = random.randint(0, 2**32)
        random.setstate(state)

    if request.module is not None:
        if request.module not in config.model.premodules and request.module != "vanilla":
            return False, "module should be one of: " + ", ".join(config.model.premodules)

    max_gens = 100
    if 0:
        num_gen_tiers = [(1024*512, 4), (640*640, 6), (704*512, 8), (512*512, 16), (384*640, 18)]
        pixel_count = request.width * request.height
        for tier in num_gen_tiers:
            if pixel_count <= tier[0]:
                max_gens = tier[1]
            else:
                break
    if request.n_samples > max_gens:
        return False, f"requested more ({request.n_samples}) images than possible at this resolution"

    if request.image is not None:
        #decode from base64
        try:
            request.image = base64.b64decode(request.image.encode('utf-8'))

        except Exception as e:
            traceback.print_exc()
            return False, "image is not valid base64"
        #check if image is valid
        try:
            from PIL import Image
            image = Image.open(BytesIO(request.image))
            image.verify()

        except Exception as e:
            traceback.print_exc()
            return False, "image is not valid"

        #image is valid, load it again(still check again, verify() can't be sure as it doesn't decode.)
        try:
            image = Image.open(BytesIO(request.image))
            image = image.convert('RGB')
            image = image.resize((request.width, request.height), resample=Image.Resampling.LANCZOS)
            request.image = image
        except Exception as e:
            traceback.print_exc()
            return False, "Error while opening and cleaning image"

    if request.masks is not None:
        masks = request.masks
        for x in range(len(masks)):
            image = masks[x]["mask"]
            try:
                image_bytes = base64.b64decode(image.encode('utf-8'))

            except Exception as e:
                traceback.print_exc()
                return False, "image is not valid base64"

            try:
                from PIL import Image
                image = Image.open(BytesIO(image_bytes))
                image.verify()

            except Exception as e:
                traceback.print_exc()
                return False, "image is not valid"

            #image is valid, load it again(still check again, verify() can't be sure as it doesn't decode.)
            try:
                image = Image.open(BytesIO(image_bytes))
                #image = image.convert('RGB')
                image = image.resize((request.width//request.downsampling_factor, request.height//request.downsampling_factor), resample=Image.Resampling.LANCZOS)

            except Exception as e:
                traceback.print_exc()
                return False, "Error while opening and cleaning image"

            masks[x]["mask"] = image

    return True, request

def sanitize_dalle_mini(request):
    return True, request

def sanitize_basedformer(request):
    return True, request

def sanitize_embedder(request):
    return True, request

def sanitize_input(config, request):
    """
    Sanitize the input data and set defaults
    """
    request = DotMap(request)
    default, forced_default = defaults[config.model_name]
    for k, v in default.items():
        if k not in request:
            request[k] = v

    for k, v in forced_default.items():
        request[k] = v

    if config.model_name == 'stable-diffusion':
        return sanitize_stable_diffusion(request, config)

    elif config.model_name == 'dalle-mini':
        return sanitize_dalle_mini(request)

    elif config.model_name == 'basedformer':
        return sanitize_basedformer(request)

    elif config.model_name == "embedder":
        return sanitize_embedder(request)
