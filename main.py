import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import httpx

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v2 as imageio

import fal_client

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

fal_client.api_key = os.getenv("FAL_KEY")

OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    text: str
    duration: int = 5
    format: str = "9:16"
    cfg: float = 0.5

# =========================
# HELPERS
# =========================
def extract_video_url(result: dict):
    if not isinstance(result, dict):
        return None
    if result.get("video") and result["video"].get("url"):
        return result["video"]["url"]
    if result.get("videos"):
        return result["videos"][0].get("url")
    if result.get("output") and result["output"].get("url"):
        return result["output"]["url"]
    return result.get("url")

# =========================
# ROOT / HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

# =========================
# UI
# =========================
@app.get("/app", response_class=HTMLResponse)
def app_ui():
    return open("ui.html").read() if os.path.exists("ui.html") else "<h1>UI cargada</h1>"

# =========================
# CHAT
# =========================
@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "Eres amable, empático y claro."},
                    {"role": "user", "content": req.message},
                ],
            },
        )
    return {"reply": r.json()["choices"][0]["message"]["content"]}

# =========================
# IMAGE
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": req.prompt,
                "size": "1024x1024",
            },
        )
    return r.json()

# =========================
# REELS MP4 (LOCAL)
# =========================
def make_reels(text: str, duration: int, out_path: Path):
    w, h = 720, 1280
    fps = 24
    frames = duration * fps

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps)

    for i in range(frames):
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)
        draw.text((60, h//2), text, font=font, fill=(255, 255, 255))
        writer.append_data(np.array(img))

    writer.close()

@app.post("/reels")
def reels(req: VideoRequest):
    name = f"reels_{uuid.uuid4().hex}.mp4"
    path = OUT_DIR / name
    make_reels(req.text, req.duration, path)
    return FileResponse(path, media_type="video/mp4", filename=name)

# =========================
# 🎬 VIDEO CINE (TEXTO → VIDEO)
# =========================
@app.post("/video-cine")
def video_cine(req: VideoRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}

    try:
        result = fal_client.run(
            "runwayml/gen2",
            arguments={
                "prompt": req.text,
                "seconds": req.duration,
            },
        )
        url = extract_video_url(result)
        if not url:
            return {"status": "error", "message": "No se obtuvo video_url", "raw": result}
        return {"status": "ok", "video_url": url}
    except Exception as e:
        return {"status": "error", "message": str(e)}
