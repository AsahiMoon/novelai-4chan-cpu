import os
import re
import sys
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from hydra_node.config import init_config_model
from hydra_node.models import EmbedderModel
from typing import Optional, List
from typing_extensions import TypedDict
import socket
from hydra_node.sanitize import sanitize_input
import uvicorn
from typing import Union
import time
import gc
import io
import signal
import base64
import traceback
import threading
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

genlock = threading.Lock()

TOKEN = os.getenv("TOKEN", None)
print(f"Starting Hydra Node HTTP TOKEN={TOKEN}")

#Initialize model and config
model, config, model_hash = init_config_model()
try:
    embedmodel = EmbedderModel()
except Exception as e:
    print("couldn't load embed model, suggestions won't work:", e)
    embedmodel = False
logger = config.logger
try:
    config.mainpid = int(open("gunicorn.pid", "r").read())
except FileNotFoundError:
    config.mainpid = os.getpid()
mainpid = config.mainpid
hostname = socket.gethostname()
sent_first_message = False

def verify_token(req: Request):
    if TOKEN:
        valid = "Authorization" in req.headers and req.headers["Authorization"] == "Bearer "+TOKEN
        if not valid:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
    return True
#Initialize fastapi
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("FastAPI Started, serving")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("FastAPI Shutdown, exiting")

class Masker(TypedDict):
    seed: int
    mask: str

class Tags(TypedDict):
    tag: str
    count: int
    confidence: float

class GenerationRequest(BaseModel):
    prompt: str
    image: str = None
    n_samples: int = 1
    steps: int = 50
    sampler: str = "plms"
    fixed_code: bool = False
    ddim_eta: float = 0.0
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downsampling_factor: int = 8
    scale: float = 7.0
    dynamic_threshold: float = None
    seed: int = None
    temp: float = 1.0
    top_k: int = 256
    grid_size: int = 4
    advanced: bool = False
    stage_two_seed: int = None
    strength: float = 0.69
    noise: float = 0.667
    mitigate: bool = False
    module: str = None
    masks: List[Masker] = None
    uc: str = None

class TextRequest(BaseModel):
    prompt: str

class TagOutput(BaseModel):
    tags: List[Tags]

class TextOutput(BaseModel):
    is_safe: str
    corrected_text: str

class GenerationOutput(BaseModel):
    output: List[str]

class ErrorOutput(BaseModel):
    error: str

def saveimage(image, request):
    os.makedirs("images", exist_ok=True)

    filename = request.prompt.replace('masterpiece, best quality, ', '')
    filename = re.sub(r'[/\\<>:"|]', '', filename)
    filename = filename[:128]
    filename += f' s-{request.seed}'
    filename = os.path.join("images", filename.strip())

    for n in range(1000000):
        suff = '.png'
        if n:
            suff = f'-{n}.png'
        if not os.path.exists(filename + suff):
            break

    try:
        with open(filename + suff, "wb") as f:
            f.write(image)
    except Exception as e:
        print("failed to save image:", e)

@app.post('/generate-stream')
def generate(request: GenerationRequest, authorized: bool = Depends(verify_token)):
    t = time.perf_counter()
    try:
        output = sanitize_input(config, request)

        if output[0]:
            request = output[1]
        else:
            return ErrorOutput(error=output[1])

        with genlock:
            if request.advanced:
                if request.n_samples > 1:
                    return ErrorOutput(error="advanced mode does not support n_samples > 1")

                images = model.sample_two_stages(request)
            else:
                images = model.sample(request)

        seed = request.seed

        images_encoded = []
        for x in range(len(images)):
            if seed is not None:
                request.seed = seed
                seed += 1
            comment = json.dumps({"steps":request.steps,"sampler":request.sampler,"seed":request.seed,"strength":request.strength,"noise":request.noise,"scale":request.scale,"uc":request.uc})
            metadata = PngInfo()
            metadata.add_text("Title", "AI generated image")
            metadata.add_text("Description", request.prompt)
            metadata.add_text("Software", "NovelAI")
            metadata.add_text("Source", "Stable Diffusion "+model_hash)
            metadata.add_text("Comment", comment)
            image = Image.fromarray(images[x])
            #save pillow image with bytesIO
            output = io.BytesIO()
            image.save(output, format='PNG', pnginfo=metadata)
            image = output.getvalue()
            if config.savefiles:
                saveimage(image, request)
            #get base64 of image
            image = base64.b64encode(image).decode("ascii")
            images_encoded.append(image)


        del images

        process_time = time.perf_counter() - t
        logger.info(f"Request took {process_time:0.3f} seconds")
        data = ""
        ptr = 0
        for x in images_encoded:
            ptr += 1
            data += ("event: newImage\nid: {}\ndata:{}\n\n").format(ptr, x)
        return Response(content=data, media_type="text/event-stream")
        #return GenerationOutput(output=images)

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        e_s = str(e)
        gc.collect()
        if "CUDA out of memory" in e_s or \
                "an illegal memory access" in e_s or "CUDA" in e_s:
            logger.error("GPU error, committing seppuku.")
            os.kill(mainpid, signal.SIGTERM)
        return {"error": str(e)}

@app.post('/generate', response_model=Union[GenerationOutput, ErrorOutput])
def generate(request: GenerationRequest, authorized: bool = Depends(verify_token)):
    t = time.perf_counter()
    try:
        output = sanitize_input(config, request)

        if output[0]:
            request = output[1]
        else:
            return ErrorOutput(error=output[1])

        with genlock:
            images = model.sample(request)
        images_encoded = []
        for x in range(len(images)):
            image = Image.fromarray(images[x])
            comment = json.dumps({"steps":request.steps,"sampler":request.sampler,"seed":request.seed,"strength":request.strength,"noise":request.noise,"scale":request.scale,"uc":request.uc})
            metadata = PngInfo()
            metadata.add_text("Title", "AI generated image")
            metadata.add_text("Description", request.prompt)
            metadata.add_text("Software", "NovelAI")
            metadata.add_text("Source", "Stable Diffusion "+model_hash)
            metadata.add_text("Comment", comment)
            image = Image.fromarray(images[x])
            #save pillow image with bytesIO
            output = io.BytesIO()
            image.save(output, format='PNG', pnginfo=metadata)
            image = output.getvalue()
            if config.savefiles:
                saveimage(image, request)
            #get base64 of image
            image = base64.b64encode(image).decode("ascii")
            images_encoded.append(image)

        del images

        process_time = time.perf_counter() - t
        logger.info(f"Request took {process_time:0.3f} seconds")
        return GenerationOutput(output=images_encoded)

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        e_s = str(e)
        gc.collect()
        if "CUDA out of memory" in e_s or \
                "an illegal memory access" in e_s or "CUDA" in e_s:
            logger.error("GPU error, committing seppuku.")
            os.kill(mainpid, signal.SIGTERM)
        return {"error": str(e)}

@app.post('/generate-text', response_model=Union[TextOutput, ErrorOutput])
def generate_text(request: TextRequest, authorized: bool = Depends(verify_token)):
    t = time.perf_counter()
    try:
        output = sanitize_input(config, request)
        if output[0]:
            request = output[1]
        else:
            return ErrorOutput(error=output[1])

        is_safe, corrected_text = model.sample(request)

        process_time = time.perf_counter() - t
        logger.info(f"Request took {process_time:0.3f} seconds")
        return TextOutput(is_safe=is_safe, corrected_text=corrected_text)

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        e_s = str(e)
        gc.collect()
        if "CUDA out of memory" in e_s or \
                "an illegal memory access" in e_s or "CUDA" in e_s:
            logger.error("GPU error, committing seppuku.")
            os.kill(mainpid, signal.SIGTERM)
        return ErrorOutput(error=str(e))

@app.get('/predict-tags', response_model=Union[TagOutput, ErrorOutput])
async def predict_tags(prompt="", authorized: bool = Depends(verify_token)):
    t = time.perf_counter()
    try:
        #output = sanitize_input(config, request)
        #if output[0]:
        #    request = output[1]
        #else:
        #    return ErrorOutput(error=output[1])

        tags = embedmodel.get_top_k(prompt)

        process_time = time.perf_counter() - t
        logger.info(f"Request took {process_time:0.3f} seconds")
        return TagOutput(tags=[Tags(tag=tag, count=count, confidence=confidence) for tag, count, confidence in tags])

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        e_s = str(e)
        gc.collect()
        if "CUDA out of memory" in e_s or \
                "an illegal memory access" in e_s or "CUDA" in e_s:
            logger.error("GPU error, committing seppuku.")
            os.kill(mainpid, signal.SIGTERM)
        return ErrorOutput(error=str(e))

@app.get('/')
def index():
    return FileResponse('static/index.html')

app.mount("/", StaticFiles(directory="static/"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4315, log_level="info")
