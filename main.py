import os
import uuid
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import fal_client

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
FAL_KEY = os.getenv("FAL_KEY")

BASE_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    mode: str  # reels | runway | pika
    text: str = ""
    duration: int = 6
    format: str = "9:16"

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

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
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente amable, profesional y empático. "
                            "Hablas con calidez y respeto."
                        ),
                    },
                    {"role": "user", "content": req.message},
                ],
            },
        )
    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMAGES
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
    data = r.json()
    return {"url": data["data"][0]["url"]}

# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    mode = req.mode.lower().strip()

    # -------- MODO A: REELS → MP4 REAL
    if mode == "reels":
        video_id = f"{uuid.uuid4()}.mp4"
        out_path = os.path.join(VIDEO_DIR, video_id)

        text = req.text.replace(":", "\\:").replace("'", "\\'")
        duration = max(3, min(req.duration, 15))

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=#0b1220:s=1080x1920:r=30",
            "-vf",
            f"drawtext=text='{text}':fontcolor=white:fontsize=64:"
            f"x=(w-text_w)/2:y=(h-text_h)/2",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            out_path
        ]

        subprocess.run(cmd, check=True)

        return {
            "status": "ok",
            "provider": "reels",
            "video_url": f"/videos/{video_id}"
        }

    # -------- RUNWAY / PIKA (via FAL)
    if not FAL_KEY:
        return {"error": "FAL_KEY no configurada en Render"}

    try:
        if mode == "runway":
            result = fal_client.run(
                "runwayml/gen2",
                arguments={"prompt": req.text, "seconds": req.duration},
            )
        elif mode == "pika":
            result = fal_client.run(
                "pika/video",
                arguments={"prompt": req.text, "seconds": req.duration},
            )
        else:
            return {"error": "Modo inválido"}

        video_url = None
        if isinstance(result, dict):
            video_url = (
                result.get("video", {}).get("url")
                or result.get("video_url")
                or result.get("url")
            )

        return {
            "status": "ok",
            "provider": mode,
            "video_url": video_url,
            "raw": result
        }

    except Exception as e:
        return {"error": str(e)}
