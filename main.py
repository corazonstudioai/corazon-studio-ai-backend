import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
PIKA_API_KEY = os.getenv("PIKA_API_KEY")

# Carpeta para guardar mp4
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
    mode: str  # reels | runway | pika
    text: str = ""
    duration: int = 6
    format: str = "9:16"  # "9:16" o "16:9"

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

# =========================
# CHAT (AMABLE + EMPÁTICO)
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
                            "Hablas con calidez, respeto y claridad. Ayudas con paciencia."
                        ),
                    },
                    {"role": "user", "content": req.message},
                ],
            },
        )
    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMÁGENES
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
# HELPERS: REELS MP4
# =========================
def _reels_size(fmt: str):
    # 9:16 vertical o 16:9 horizontal
    if fmt.strip() == "16:9":
        return (1280, 720)
    return (720, 1280)

def _wrap_text(text: str, max_chars: int = 22):
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines[:6]

def _make_reels_mp4(text: str, duration: int, fmt: str, out_path: Path):
    w, h = _reels_size(fmt)
    fps = 24
    total_frames = max(1, duration * fps)

    # Fuente simple (fallback seguro)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    lines = _wrap_text(text, max_chars=22)

    for i in range(total_frames):
        # Fondo degradado simple
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)

        # Animación: subir suave el bloque de texto
        t = i / max(1, total_frames - 1)
        y_base = int(h * (0.62 - 0.08 * (1 - np.cos(np.pi * t)) / 2))

        # Caja semi-transparente simulada
        box_margin = 60
        box_w = w - 2 * box_margin
        box_h = 70 * len(lines) + 80
        x0 = box_margin
        y0 = y_base - box_h // 2
        x1 = x0 + box_w
        y1 = y0 + box_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(18, 27, 46))

        # Texto centrado
        y = y0 + 40
        for ln in lines:
            tw = draw.textlength(ln, font=font)
            tx = (w - tw) / 2
            draw.text((tx, y), ln, font=font, fill=(255, 255, 255))
            y += 70

        # Convertir a frame
        frame = np.array(img)
        writer.append_data(frame)

    writer.close()

# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    mode = (req.mode or "").strip().lower()

    # -------- MODO A: REELS (MP4 DESCARGABLE)
    if mode == "reels":
        out_name = f"reels_{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name

        _make_reels_mp4(req.text or " ", max(1, int(req.duration)), req.format, out_path)

        # Devuelve archivo mp4 para descargar
        return FileResponse(
            path=str(out_path),
            media_type="video/mp4",
            filename=out_name,
        )

    # -------- MODO B: RUNWAY (via FAL)
    if mode == "runway":
        # Nota: fal_client usa FAL_KEY en variables de entorno
        result = fal_client.run(
            "runwayml/gen2",
            arguments={
                "prompt": req.text,
                "seconds": req.duration,
            },
        )
        return {"status": "ok", "provider": "runway", "result": result}

    # -------- MODO C: PIKA (via FAL)
    if mode == "pika":
        result = fal_client.run(
            "pika/video",
            arguments={
                "prompt": req.text,
                "seconds": req.duration,
            },
        )
        return {"status": "ok", "provider": "pika", "result": result}

    return {"error": "Modo de video no válido. Usa: reels | runway | pika"}
