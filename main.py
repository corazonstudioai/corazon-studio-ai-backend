import os
import uuid
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

# Para generar MP4 (sin instalar ffmpeg manualmente)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import fal_client

# =========================
# CONFIG (ENV VARS)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

# Nota: fal_client usa FAL_KEY. Si tienes Runway/Pika, normalmente lo manejas por fal.
FAL_KEY = os.getenv("FAL_KEY", "").strip()
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "").strip()
PIKA_API_KEY = os.getenv("PIKA_API_KEY", "").strip()

# Carpeta temporal para MP4 en Render
TMP_DIR = "/tmp/corazonstudio"
os.makedirs(TMP_DIR, exist_ok=True)

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
    format: str = "9:16"  # 9:16 | 1:1 | 16:9


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
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en Render (Environment).")

    system_msg = (
        "Eres un asistente amable, profesional y empático. "
        "Hablas con calidez, respeto y claridad. Ayudas con paciencia. "
        "Respondes en español a menos que el usuario pida otro idioma."
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        raise HTTPException(status_code=500, detail=f"Error OpenAI chat: {err}")

    data = r.json()
    try:
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Respuesta inesperada de OpenAI: {data}")

    return {"reply": reply}


# =========================
# IMÁGENES (devuelve base64)
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en Render (Environment).")

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": req.prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        raise HTTPException(status_code=500, detail=f"Error OpenAI image: {err}")

    data = r.json()
    try:
        b64 = data["data"][0]["b64_json"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Respuesta inesperada de OpenAI: {data}")

    return {"status": "ok", "b64": b64}


# =========================
# FILES (para descargar MP4)
# =========================
@app.get("/files/{filename}")
def get_file(filename: str):
    path = os.path.join(TMP_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    # Render permite servir desde /tmp dentro del contenedor
    return FileResponse(path, media_type="video/mp4", filename=filename)


# =========================
# REELS -> MP4 GENERATOR
# =========================
def _pick_size(format_str: str):
    # tamaños razonables
    if format_str == "9:16":
        return (720, 1280)
    if format_str == "16:9":
        return (1280, 720)
    return (1080, 1080)  # 1:1

def _wrap_text(text: str, max_chars: int = 18):
    words = text.split()
    lines = []
    line = []
    count = 0
    for w in words:
        if count + len(w) + (1 if line else 0) <= max_chars:
            line.append(w)
            count += len(w) + (1 if line else 0)
        else:
            lines.append(" ".join(line))
            line = [w]
            count = len(w)
    if line:
        lines.append(" ".join(line))
    return lines[:8]  # evita demasiado texto

def generate_reels_mp4(text: str, duration: int, format_str: str) -> str:
    w, h = _pick_size(format_str)
    fps = 24
    total_frames = max(1, int(duration * fps))

    # Fuente: PIL intentará cargar una por defecto si no hay TTF
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=int(h * 0.05))
        font_bold = ImageFont.truetype("DejaVuSans.ttf", size=int(h * 0.065))
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    lines = _wrap_text(text.strip() if text.strip() else "Escribe tu texto del reels…")

    # MP4 path
    file_id = str(uuid.uuid4())
    filename = f"reels_{file_id}.mp4"
    out_path = os.path.join(TMP_DIR, filename)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

    for i in range(total_frames):
        t = i / max(1, total_frames - 1)  # 0..1

        # Fondo con gradiente simple
        bg = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(bg)

        # brillo suave animado
        glow_y = int(h * (0.2 + 0.6 * t))
        glow_r = int(min(w, h) * 0.55)
        for r in range(glow_r, 0, -40):
            alpha = int(35 * (r / glow_r))
            color = (20 + int(40*t), 80 + int(50*t), 140 + int(60*t))
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.ellipse(
                (w//2 - r, glow_y - r, w//2 + r, glow_y + r),
                fill=(color[0], color[1], color[2], alpha),
            )
            bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(bg)

        # Panel
        pad = int(w * 0.08)
        panel_top = int(h * 0.22)
        panel_bottom = int(h * 0.78)
        draw.rounded_rectangle(
            (pad, panel_top, w - pad, panel_bottom),
            radius=28,
            fill=(18, 26, 44),
        )

        # Animación: aparición de texto (slide + fade)
        # progreso de texto
        appear = min(1.0, max(0.0, (t - 0.08) / 0.25))
        y_offset = int((1.0 - appear) * 50)

        # Título
        title = "Reels"
        tw, th = draw.textbbox((0, 0), title, font=font_bold)[2:]
        draw.text(((w - tw) // 2, panel_top + int(h*0.04) + y_offset), title, font=font_bold, fill=(230, 240, 255))

        # Texto centrado
        start_y = panel_top + int(h * 0.14) + y_offset
        line_gap = int(h * 0.07)

        for idx, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (w - lw) // 2
            y = start_y + idx * line_gap
            # sombra
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=(235, 245, 255))

        # Footer
        footer = "Corazón Studio AI"
        fb = draw.textbbox((0, 0), footer, font=font)[2:]
        draw.text(((w - fb[0]) // 2, panel_bottom - int(h*0.08)), footer, font=font, fill=(170, 190, 220))

        frame = np.array(bg)
        writer.append_data(frame)

    writer.close()
    return filename


# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    mode = (req.mode or "").strip().lower()
    duration = int(req.duration) if req.duration else 6
    duration = max(2, min(duration, 20))  # limites razonables
    format_str = (req.format or "9:16").strip()

    # -------- MODO A: REELS -> MP4 descargable
    if mode == "reels":
        filename = generate_reels_mp4(req.text, duration, format_str)
        base_url = os.getenv("PUBLIC_BASE_URL", "").strip()
        # Si no defines PUBLIC_BASE_URL, el frontend ya sabe tu backend y arma la URL.
        return {"status": "ok", "provider": "reels", "filename": filename}

    # -------- MODO B: RUNWAY (vía fal_client)
    if mode == "runway":
        if not os.getenv("FAL_KEY") and not FAL_KEY:
            raise HTTPException(status_code=500, detail="Falta FAL_KEY en Render (Environment) para runway/pika.")
        result = fal_client.run(
            "runwayml/gen2",
            arguments={"prompt": req.text, "seconds": duration},
        )
        return {"status": "ok", "provider": "runway", "result": result}

    # -------- MODO C: PIKA (vía fal_client)
    if mode == "pika":
        if not os.getenv("FAL_KEY") and not FAL_KEY:
            raise HTTPException(status_code=500, detail="Falta FAL_KEY en Render (Environment) para runway/pika.")
        result = fal_client.run(
            "pika/video",
            arguments={"prompt": req.text, "seconds": duration},
        )
        return {"status": "ok", "provider": "pika", "result": result}

    return {"error": "Modo de video no válido. Usa reels, runway o pika."}
