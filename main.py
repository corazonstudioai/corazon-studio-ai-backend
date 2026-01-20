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

from fastapi import FastAPI, Request
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
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

# FAL
fal_client.api_key = os.getenv("FAL_KEY")  # debe ser EXACTO: FAL_KEY en Render

# FFmpeg
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Música de fondo (opcional): agrega estos archivos a tu repo:
# assets/music_soft.mp3
# assets/music_cinematic.mp3
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
    instructions: Optional[str] = None

class ReelsVoiceRequest(BaseModel):
    text: str
    duration: int = 6
    format: str = "9:16"
    language: Literal["es", "en"] = "es"
    mode: Literal["narrator_only", "narrator_plus_character"] = "narrator_only"
    voice_gender: Literal["female", "male"] = "female"
    music: Literal["none", "soft", "cinematic"] = "soft"

class CineVoiceRequest(BaseModel):
    text: str
    duration: int = 5
    format: str = "9:16"
    cfg: float = 0.5
    language: Literal["es", "en"] = "es"
    mode: Literal["narrator_only", "narrator_plus_character"] = "narrator_only"
    voice_gender: Literal["female", "male"] = "female"
    music: Literal["none", "soft", "cinematic"] = "soft"

# =========================
# HELPERS
# =========================
def run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {p.stderr[-1500:]}")

def safe_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext.lstrip('.')}"

def file_path(name: str) -> Path:
    return OUT_DIR / name

def absolute_url(req: Request, path: str) -> str:
    # path debe empezar con "/"
    base = str(req.base_url).rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path

def make_reels_video(text: str, duration: int, out_path: Path):
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

async def openai_tts_to_file(
    text: str,
    voice: str,
    out_path: Path,
    speed: float = 1.0,
    response_format: str = "mp3",
    instructions: Optional[str] = None,
):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")

    payload = {
        "model": OPENAI_TTS_MODEL,
        "input": text[:4096],
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

def pick_voices(language: str, voice_gender: str) -> tuple[str, str]:
    """
    (voz_narrador, voz_personaje)
    Usamos voces distintas para que se note el personaje.
    """
    if language == "es":
        if voice_gender == "male":
            return ("onyx", "alloy")
        return ("nova", "shimmer")
    else:
        if voice_gender == "male":
            return ("onyx", "echo")
        return ("nova", "fable")

async def tts_narrator_and_character(
    language: str,
    mode: str,
    voice_gender: str,
    text: str,
) -> tuple[Path, Optional[Path]]:
    narr_voice, char_voice = pick_voices(language, voice_gender)

    narr_path = file_path(f"narr_{uuid.uuid4().hex}.mp3")
    char_path = file_path(f"char_{uuid.uuid4().hex}.mp3")

    if language == "es":
        narr_text = text
        char_text = "¡Hey! Estoy aquí contigo. Dios no se olvida de ti."
        narr_instr = "Español latino neutro. Voz cálida, clara, ritmo natural."
        char_instr = "Español latino neutro. Voz más expresiva y alegre, como personaje."
    else:
        narr_text = text
        char_text = "Hey! I'm here with you. God hasn't forgotten you."
        narr_instr = "Clear English. Warm narrator, natural pacing."
        char_instr = "Clear English. More expressive character voice."

    await openai_tts_to_file(narr_text, voice=narr_voice, out_path=narr_path, response_format="mp3", instructions=narr_instr)

    if mode == "narrator_plus_character":
        await openai_tts_to_file(char_text, voice=char_voice, out_path=char_path, response_format="mp3", instructions=char_instr)
        return narr_path, char_path

    return narr_path, None

def concat_mp3(a: Path, b: Path, out_mp3: Path):
    """
    Une 2 mp3 en 1 mp3 (re-encode) usando concat demuxer.
    """
    list_file = file_path(f"list_{uuid.uuid4().hex}.txt")
    # Importante: usar rutas POSIX
    list_file.write_text(f"file '{a.as_posix()}'\nfile '{b.as_posix()}'\n", encoding="utf-8")

    run_ffmpeg([
        FFMPEG_PATH, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame",
        str(out_mp3),
    ])

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
    - voz sola
    - voz + música (ducking)
    - música sola
    - nada: copia video
    """
    if voice_path and music_path:
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

    out_path.write_bytes(video_path.read_bytes())

async def download_to_file(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url)
        r.raise_for_status()
        out_path.write_bytes(r.content)

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
async def image(req: ImageRequest, request: Request):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_IMAGE_MODEL, "prompt": req.prompt, "size": req.size},
        )
        r.raise_for_status()
        data = r.json()

    b64 = None
    try:
        b64 = data["data"][0].get("b64_json")
    except Exception:
        b64 = None

    if not b64:
        return {"status": "ok", "raw": data}

    img_bytes = base64.b64decode(b64)
    name = safe_filename("png")
    path = file_path(name)
    path.write_bytes(img_bytes)

    rel = f"/files/{name}"
    return {"status": "ok", "image_url": rel, "image_url_full": absolute_url(request, rel)}

# =========================
# REELS MP4 (LOCAL)
# =========================
@app.post("/reels")
def reels(req: VideoRequest):
    name = f"reels_{uuid.uuid4().hex}.mp4"
    path = file_path(name)
    make_reels_video(req.text, req.duration, path)
    return FileResponse(path, media_type="video/mp4", filename=name)

# =========================
# TTS -> mp3 servido por /files/...
# =========================
@app.post("/tts")
async def tts(req: TTSRequest, request: Request):
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

    rel = f"/files/{name}"
    return {"status": "ok", "audio_url": rel, "audio_url_full": absolute_url(request, rel)}

# =========================
# REELS CON VOZ + MÚSICA (con personaje real)
# =========================
@app.post("/reels-voice")
async def reels_voice(req: ReelsVoiceRequest, request: Request):
    # 1) video base
    base_name = f"reels_{uuid.uuid4().hex}.mp4"
    base_path = file_path(base_name)
    make_reels_video(req.text, req.duration, base_path)

    # 2) voz: narrador + personaje (2 voces) si aplica
    narr_mp3, char_mp3 = await tts_narrator_and_character(req.language, req.mode, req.voice_gender, req.text)

    voice_path = narr_mp3
    if char_mp3:
        merged = file_path(f"voice_{uuid.uuid4().hex}.mp3")
        concat_mp3(narr_mp3, char_mp3, merged)
        voice_path = merged

    # 3) música
    music_path = choose_music_path(req.music)

    # 4) mux final
    out_name = f"reels_voice_{uuid.uuid4().hex}.mp4"
    out_path = file_path(out_name)
    mux_video_audio(base_path, voice_path, music_path, out_path)

    rel = f"/files/{out_name}"
    return {"status": "ok", "video_url": rel, "video_url_full": absolute_url(request, rel)}

# =========================
# VIDEO CINE (TEXTO → VIDEO) (FAL)
# =========================
@app.post("/video-cine")
def video_cine(req: VideoRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}

    try:
        result = fal_client.run(
            os.getenv("VIDEO_MODEL_CINE", "fal-ai/wan-v2.1/14b/image-to-video"),
            arguments={
                "prompt": req.text,
                "duration": req.duration,
            },
        )

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
# VIDEO CINE + VOZ + MUSICA (con personaje real)
# =========================
@app.post("/video-cine-voice")
async def video_cine_voice(req: CineVoiceRequest, request: Request):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no configurada"}

    try:
        # 1) generar cine con tu modelo actual
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

        # 4) voz narrador + personaje (2 voces)
        narr_mp3, char_mp3 = await tts_narrator_and_character(req.language, req.mode, req.voice_gender, req.text)
        voice_path = narr_mp3
        if char_mp3:
            merged = file_path(f"voice_{uuid.uuid4().hex}.mp3")
            concat_mp3(narr_mp3, char_mp3, merged)
            voice_path = merged

        # 5) música
        music_path = choose_music_path(req.music)

        # 6) mux final
        out_name = f"cine_voice_{uuid.uuid4().hex}.mp4"
        out_path = file_path(out_name)
        mux_video_audio(base_path, voice_path, music_path, out_path)

        rel = f"/files/{out_name}"
        return {"status": "ok", "video_url": rel, "video_url_full": absolute_url(request, rel)}

    except Exception as e:
        return {"status": "error", "message": str(e)}
