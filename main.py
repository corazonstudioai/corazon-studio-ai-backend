import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import fal_client


# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# Si usas fal para Runway/Pika, necesitas FAL_KEY en Render
FAL_KEY = os.getenv("FAL_KEY")


# Carpeta para archivos (videos mp4)
FILES_DIR = Path("/tmp/files")
FILES_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok para demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos generados (mp4)
app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")


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
# HELPERS
# =========================
def _require_env(name: str, value: str | None):
    if not value:
        return False, f"Falta variable de entorno: {name}"
    return True, ""


def _make_reels_mp4(text: str, duration: int, out_path: Path) -> None:
    """
    Genera un MP4 simple tipo Reels:
    - Fondo oscuro
    - Texto centrado
    - Animación: aparece por líneas con un leve "fade" simulado por brillo
    """
    duration = max(2, min(int(duration), 15))  # 2..15 seg
    fps = 24
    total_frames = duration * fps

    # 9:16 base 720x1280 (ligero para Render)
    W, H = 720, 1280

    # Tipografía (PIL default si no hay fuente)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 52)
    except Exception:
        font = ImageFont.load_default()

    # Preparar líneas
    raw = (text or "").strip()
    if not raw:
        raw = "Escribe tu texto para el Reels ✨"
    # Cortar en líneas medianas
    words = raw.split()
    lines = []
    line = []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 18:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    lines = lines[:6]  # máximo 6 líneas

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    for i in range(total_frames):
        t = i / total_frames  # 0..1

        # Fondo
        img = Image.new("RGB", (W, H), (10, 14, 25))
        draw = ImageDraw.Draw(img)

        # Caja estilo card
        pad = 60
        card_top = 280
        card_bottom = 1000
        draw.rounded_rectangle(
            (pad, card_top, W - pad, card_bottom),
            radius=28,
            fill=(18, 24, 42),
        )

        # Animación: vamos mostrando líneas poco a poco
        # progreso de texto (0..1)
        reveal = min(1.0, max(0.0, (t * 1.2)))
        n_show = max(1, int(reveal * len(lines)))

        shown = lines[:n_show]

        # Simular “fade” con brillo
        fade = min(1.0, (t * 2.0))
        color = (
            int(220 * fade),
            int(230 * fade),
            int(255 * fade),
        )

        # Dibujar texto centrado
        y = 420
        for ln in shown:
            bbox = draw.textbbox((0, 0), ln, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, y), ln, font=font, fill=color)
            y += 80

        # Footer pequeño
        footer = "Corazón Studio AI"
        fnt2 = ImageFont.load_default()
        bbox2 = draw.textbbox((0, 0), footer, font=fnt2)
        draw.text((W - (bbox2[2] - bbox2[0]) - 20, H - 30), footer, font=fnt2, fill=(120, 130, 160))

        writer.append_data(imageio.asarray(img))

    writer.close()


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
    ok, msg = _require_env("OPENAI_API_KEY", OPENAI_API_KEY)
    if not ok:
        return {"error": msg}

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

    if r.status_code != 200:
        return {"error": "OpenAI error", "detail": r.text}

    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}


# =========================
# IMÁGENES
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    ok, msg = _require_env("OPENAI_API_KEY", OPENAI_API_KEY)
    if not ok:
        return {"error": msg}

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

    if r.status_code != 200:
        return {"error": "OpenAI image error", "detail": r.text}

    return r.json()


# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    mode = (req.mode or "").strip().lower()

    # -------- MODO A: REELS (MP4 real)
    if mode == "reels":
        vid_id = str(uuid.uuid4())
        out_path = FILES_DIR / f"reels_{vid_id}.mp4"
        _make_reels_mp4(req.text, req.duration, out_path)

        # URL pública del archivo
        return {
            "status": "ok",
            "type": "reels",
            "video_url": f"/files/{out_path.name}",
            "message": "MP4 generado (Reels)",
        }

    # -------- MODO B: RUNWAY (via fal)
    if mode == "runway":
        if not FAL_KEY:
            return {"error": "Falta FAL_KEY en Render (para usar Runway vía fal_client)."}
        result = fal_client.run(
            "runwayml/gen2",
            arguments={
                "prompt": req.text,
                "seconds": req.duration,
            },
        )
        return {"status": "ok", "provider": "runway", "result": result}

    # -------- MODO C: PIKA (via fal)
    if mode == "pika":
        if not FAL_KEY:
            return {"error": "Falta FAL_KEY en Render (para usar Pika vía fal_client)."}
        result = fal_client.run(
            "pika/video",
            arguments={
                "prompt": req.text,
                "seconds": req.duration,
            },
        )
        return {"status": "ok", "provider": "pika", "result": result}

    return {"error": "Modo de video no válido. Usa: reels | runway | pika"}
