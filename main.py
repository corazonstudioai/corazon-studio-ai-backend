import os
import uuid
import base64
import subprocess
from pathlib import Path
from typing import Optional, Literal

import httpx
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import fal_client

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# TTS (OpenAI)
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")  # ver docs
# Voces built-in: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse, marin, cedar
# (según API reference) 1

# FAL
fal_client.api_key = os.getenv("FAL_KEY")  # debe ser EXACTO: FAL_KEY en Render

# FFmpeg (Render suele tener ffmpeg; si no, setea FFMPEG_PATH)
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Música de fondo (opcional): agrega estos archivos a tu repo si quieres:
# /assets/music_soft.mp3
# /assets/music_cinematic.mp3
ASSETS_DIR = Path("assets")
MUSIC_SOFT = ASSETS_DIR / "music_soft.mp3"
MUSIC_CINEMATIC = ASSETS_DIR / "music_cinematic.mp3"

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
    size: str = "1024x1024"

class VideoRequest(BaseModel):
    text: str
    duration: int = 5
    format: str = "9:16"
    cfg: float = 0.5

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"
    speed: float = 1.0
    response_format: str = "mp3"
    instructions: Optional[str] = None  # opcional (no aplica para tts-1/tts-1-hd según docs) 2

class ReelsVoiceRequest(BaseModel):
    text: str
    duration: int = 6
    format: str = "9:16"
    language: Literal["es", "en"] = "es"
    mode: Literal["narrator_only", "narrator_plus_character"] = "narrator_only"
    voice_gender: Literal["female", "male"] = "female"
    music: Literal["none", "soft", "cinematic"] = "soft"

# =========================
# HELPERS
# =========================
def run_ffmpeg(cmd: list[str]) -> None:
    # Ejecuta ffmpeg y lanza error si falla
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {p.stderr[-1200:]}")

def safe_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext.lstrip('.')}"

def file_path(name: str) -> Path:
    return OUT_DIR / name

def make_reels_video(text: str, duration: int, out_path: Path):
    # video vertical simple con texto (igual a lo que ya tienes)
    w, h = 720, 1280
    fps = 24
    frames = max(1, duration * fps)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps)
    for _ in range(frames):
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)
        draw.text((60, h // 2), text, font=font, fill=(255, 255, 255))
        writer.append_data(np.array(img))
    writer.close()

async def openai_tts_to_file(text: str, voice: str, out_path: Path, speed: float = 1.0, response_format: str = "mp3", instructions: Optional[str] = None):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")

    payload = {
        "model": OPENAI_TTS_MODEL,
        "input": text[:4096],  # límite documentado 3
        "voice": voice,
        "speed": speed,
        "response_format": response_format,
    }
    if instructions:
        payload["instructions"] = instructions

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)

def choose_voice(language: str, voice_gender: str) -> str:
    # Mapeo simple. Puedes cambiarlo después.
    if voice_gender == "male":
        return "onyx"
    return "nova"

def choose_music_path(kind: str) -> Optional[Path]:
    if kind == "soft" and MUSIC_SOFT.exists():
        return MUSIC_SOFT
    if kind == "cinematic" and MUSIC_CINEMATIC.exists():
        return MUSIC_CINEMATIC
    return None

def mux_video_audio(
    video_path: Path,
    voice_path: Optional[Path],
    music_path: Optional[Path],
    out_path: Path,
):
    """
    Mezcla:
    - si solo hay voz: la pone como audio
    - si hay voz + música: mezcla con ducking suave (sidechain)
    - si no hay voz y hay música: música sola
    """
    if voice_path and music_path:
        # Mezcla con ducking para que la música baje cuando habla la voz
        run_ffmpeg([
            FFMPEG_PATH, "-y",
            "-i", str(video_path),
            "-i", str(voice_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            "[2:a]volume=0.25[m];[m][1:a]sidechaincompress=threshold=0.03:ratio=10:attack=20:release=250[mduck];[1:a][mduck]amix=inputs=2:duration=shortest[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            str(out_path),
        ])
        return

    if voice_path and not music_path:
        run_ffmpeg([
            FFMPEG_PATH, "-y",
            "-i", str(video_path),
            "-i", str(voice_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            str(out_path),
        ])
        return

    if (not voice_path) and music_path:
        run_ffmpeg([
            FFMPEG_PATH, "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            str(out_path),
        ])
        return

    # nada que mezclar: copia tal cual
    out_path.write_bytes(video_path.read_bytes())

# =========================
# ROOT / FILES / UI
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

@app.get("/files/{name}")
def files(name: str):
    path = file_path(name)
    if not path.exists():
        return {"status": "error", "message": "Archivo no encontrado"}
    media = "application/octet-stream"
    if name.endswith(".mp4"):
        media = "video/mp4"
    elif name.endswith(".mp3"):
        media = "audio/mpeg"
    elif name.endswith(".png"):
        media = "image/png"
    return FileResponse(path, media_type=media, filename=name)

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
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "Eres amable, empático y claro."},
                    {"role": "user", "content": req.message},
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMAGE (devuelve URL servida por tu backend)
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_IMAGE_MODEL, "prompt": req.prompt, "size": req.size},
        )
        r.raise_for_status()
        data = r.json()

    # gpt-image-1 suele devolver base64 en data[].b64_json (no un URL directo)
    b64 = None
    try:
        b64 = data["data"][0].get("b64_json")
    except Exception:
        b64 = None

    if not b64:
        # Devuelve lo que haya, por si tu cuenta devuelve url
        return {"status": "ok", "raw": data}

    img_bytes = base64.b64decode(b64)
    name = safe_filename("png")
    path = file_path(name)
    path.write_bytes(img_bytes)

    return {"status": "ok", "image_url": f"/files/{name}"}

# =========================
# REELS MP4 (LOCAL) - como ya lo tenías
# =========================
@app.post("/reels")
def reels(req: VideoRequest):
    name = f"reels_{uuid.uuid4().hex}.mp4"
    path = file_path(name)
    make_reels_video(req.text, req.duration, path)
    return FileResponse(path, media_type="video/mp4", filename=name)

# =========================
# TTS (OpenAI) -> mp3 servido por /files/...
# =========================
@app.post("/tts")
async def tts(req: TTSRequest):
    name = safe_filename(req.response_format)
    out_path = file_path(name)
    await openai_tts_to_file(
        text=req.text,
        voice=req.voice,
        out_path=out_path,
        speed=req.speed,
        response_format=req.response_format,
        instructions=req.instructions,
    )
    return {"status": "ok", "audio_url": f"/files/{name}"}

# =========================
# REELS CON VOZ + MÚSICA (NUEVO)
# =========================
@app.post("/reels-voice")
async def reels_voice(req: ReelsVoiceRequest):
    # 1) video base
    base_name = f"reels_{uuid.uuid4().hex}.mp4"
    base_path = file_path(base_name)
    make_reels_video(req.text, req.duration, base_path)

    # 2) texto para voz
    voice = choose_voice(req.language, req.voice_gender)

    if req.mode == "narrator_plus_character":
        # Versión simple: narrador + una línea de personaje (después lo hacemos multi-personaje)
        if req.language == "es":
            tts_text = f"Narrador: {req.text}\n\nPersonaje: ¡Qué bonito se ve todo hoy!"
            instr = "Voz clara, estilo narrador de cuento, cálida."
        else:
            tts_text = f"Narrator: {req.text}\n\nCharacter: Wow, everything looks beautiful today!"
            instr = "Clear voice, story narrator style, warm."
    else:
        tts_text = req.text
        instr = "Voz clara, cálida, ritmo natural." if req.language == "es" else "Clear, warm voice, natural pacing."

    voice_name = safe_filename("mp3")
    voice_path = file_path(voice_name)
    await openai_tts_to_file(tts_text, voice=voice, out_path=voice_path, response_format="mp3", instructions=instr)

    # 3) música
    music_path = choose_music_path(req.music)

    # 4) mux final
    out_name = f"reels_voice_{uuid.uuid4().hex}.mp4"
    out_path = file_path(out_name)
    mux_video_audio(base_path, voice_path, music_path, out_path)

    return {"status": "ok", "video_url": f"/files/{out_name}"}

# =========================
# VIDEO CINE (TEXTO → VIDEO) - (tu ya lo tienes funcionando con FAL)
# OJO: aquí NO fuerzo un modelo específico porque tú ya lograste “Video listo ✅”.
# Si quieres, luego lo dejamos fijo a tu modelo de Kling exacto.
# =========================
@app.post("/video-cine")
def video_cine(req: VideoRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}

    try:
        # IMPORTANTE:
        # Usa aquí EL MISMO modelo que a ti te dio “Video listo ✅”.
        # Ejemplo (si usas WAN en FAL): "fal-ai/wan-v2.1/14b/image-to-video" 4
        # Si usas Kling, pon el ID exacto de tu modelo Kling de FAL (el que aparece en tu UI de FAL).
        result = fal_client.run(
            "fal-ai/wan-v2.1/14b/image-to-video",
            arguments={
                "prompt": req.text,
                "duration": req.duration,
            },
        )
        # Algunas respuestas traen "video" -> "url"
        url = None
        if isinstance(result, dict):
            if result.get("video") and isinstance(result["video"], dict):
                url = result["video"].get("url")
            if not url and result.get("videos"):
                url = result["videos"][0].get("url")

        if not url:
            return {"status": "error", "message": "No se obtuvo video_url", "raw": result}
        return {"status": "ok", "video_url": url}
    except Exception as e:
        return {"status": "error", "message": str(e)}
        # =========================
# VIDEO CINE + VOZ + MUSICA (NUEVO)
# =========================
from typing import Literal

class CineVoiceRequest(BaseModel):
    text: str
    duration: int = 5
    format: str = "9:16"
    cfg: float = 0.5

    language: Literal["es", "en"] = "es"
    mode: Literal["narrator_only", "narrator_plus_character"] = "narrator_only"
    voice_gender: Literal["female", "male"] = "female"
    music: Literal["none", "soft", "cinematic"] = "soft"


async def download_to_file(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url)
        r.raise_for_status()
        out_path.write_bytes(r.content)


def build_tts_text(text: str, language: str, mode: str) -> tuple[str, str]:
    """
    Devuelve (tts_text, instructions)
    """
    if mode == "narrator_plus_character":
        if language == "es":
            tts_text = f"Narrador: {text}\n\nPersonaje: ¡Wow! Eso se ve increíble."
            instr = "Habla en español latino neutro. Voz cálida, clara, estilo narrador de cuento."
        else:
            tts_text = f"Narrator: {text}\n\nCharacter: Wow! That looks amazing."
            instr = "Speak in clear English. Warm storyteller narrator style."
    else:
        tts_text = text
        instr = "Habla en español latino neutro. Voz cálida y natural." if language == "es" else "Clear English, warm and natural pacing."
    return tts_text, instr


@app.post("/video-cine-voice")
async def video_cine_voice(req: CineVoiceRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no configurada"}

    try:
        # 1) Generar cine con el MISMO modelo que ya te funciona.
        # Si ya tienes variable OPENAI_VIDEO_MODEL o algo similar, NO la toco.
        # Aquí usa tu modelo actual (ajústalo si tu main.py usa otro ID).
        result = fal_client.run(
            os.getenv("VIDEO_MODEL_CINE", "fal-ai/wan-v2.1/14b/image-to-video"),
            arguments={
                "prompt": req.text,
                "duration": req.duration,
            },
        )

        # 2) sacar URL del video
        video_url = None
        if isinstance(result, dict):
            if result.get("video") and isinstance(result["video"], dict):
                video_url = result["video"].get("url")
            if not video_url and result.get("videos"):
                video_url = result["videos"][0].get("url")

        if not video_url:
            return {"status": "error", "message": "No se obtuvo video_url", "raw": result}

        # 3) descargar video a /tmp/out
        base_name = f"cine_{uuid.uuid4().hex}.mp4"
        base_path = file_path(base_name)
        await download_to_file(video_url, base_path)

        # 4) generar voz
        voice = choose_voice(req.language, req.voice_gender)
        tts_text, instr = build_tts_text(req.text, req.language, req.mode)

        voice_name = safe_filename("mp3")
        voice_path = file_path(voice_name)
        await openai_tts_to_file(
            tts_text,
            voice=voice,
            out_path=voice_path,
            response_format="mp3",
            instructions=instr
        )

        # 5) música
        music_path = choose_music_path(req.music)

        # 6) mux final
        out_name = f"cine_voice_{uuid.uuid4().hex}.mp4"
        out_path = file_path(out_name)
        mux_video_audio(base_path, voice_path, music_path, out_path)

        return {"status": "ok", "video_url": f"/files/{out_name}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
