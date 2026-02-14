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
fal_client.api_key = os.getenv("FAL_KEY")  # EXACTO: FAL_KEY en Render

# Modelo cine (FAL) - TEXT TO VIDEO
VIDEO_MODEL_CINE = os.getenv(
    "VIDEO_MODEL_CINE",
    "fal-ai/wan/v2.2-a14b/text-to-video/turbo"
)

# FFmpeg
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = Path("assets")
FONTS_DIR = ASSETS_DIR / "fonts"

# Música
MUSIC_SOFT = ASSETS_DIR / "music_soft.mp3"
MUSIC_CINEMATIC = ASSETS_DIR / "music_cinematic.mp3"
MUSIC_HAPPY = ASSETS_DIR / "music_happy.mp3"

# Fuentes
FONT_INSPIRADOR = FONTS_DIR / "Poppins-SemiBold.ttf"
FONT_IMPACTANTE = FONTS_DIR / "Montserrat-ExtraBold.ttf"
FONT_MINIMAL = FONTS_DIR / "Inter-SemiBold.ttf"

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
    style: Literal["inspirador", "impactante", "minimal"] = "inspirador"
    bg: Literal["gradient", "dark", "light"] = "gradient"
    text_anim: Literal["fade", "slide", "zoom"] = "fade"
    emoji_mode: Literal["auto", "off"] = "auto"
    music: Literal["none", "soft", "cinematic", "happy"] = "soft"

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
    music: Literal["none", "soft", "cinematic", "happy"] = "soft"
    style: Literal["inspirador", "impactante", "minimal"] = "inspirador"
    bg: Literal["gradient", "dark", "light"] = "gradient"
    text_anim: Literal["fade", "slide", "zoom"] = "fade"
    emoji_mode: Literal["auto", "off"] = "auto"

class CineVoiceRequest(BaseModel):
    text: str
    duration: int = 5
    format: str = "9:16"
    cfg: float = 0.5
    language: Literal["es", "en"] = "es"
    mode: Literal["narrator_only", "narrator_plus_character"] = "narrator_only"
    voice_gender: Literal["female", "male"] = "female"
    music: Literal["none", "soft", "cinematic", "happy"] = "soft"

# =========================
# HELPERS
# =========================
def run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {p.stderr[-1600:]}")

def safe_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext.lstrip('.')}"

def file_path(name: str) -> Path:
    return OUT_DIR / name

def absolute_url(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"

def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _font_for_style(style: str, size: int) -> ImageFont.FreeTypeFont:
    if style == "impactante":
        return _load_font(FONT_IMPACTANTE, size)
    if style == "minimal":
        return _load_font(FONT_MINIMAL, size)
    return _load_font(FONT_INSPIRADOR, size)

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:8]

def _auto_emojis(text: str) -> str:
    t = text.lower()
    picks = []
    if any(k in t for k in ["dios", "fe", "cristo", "señor", "oración", "oracion"]):
        picks += ["🙏", "✨"]
    if any(k in t for k in ["bendición", "bendicion", "milagro"]):
        picks += ["🙌"]
    if any(k in t for k in ["amor", "familia", "hijo", "hija"]):
        picks += ["❤️"]
    if any(k in t for k in ["fuerza", "valiente", "ánimo", "animo", "sigue"]):
        picks += ["💪", "🔥"]
    picks = list(dict.fromkeys(picks))[:2]
    return f"{text} {' '.join(picks)}" if picks else text

def _bg_image(w: int, h: int, kind: str) -> Image.Image:
    if kind == "light":
        return Image.new("RGB", (w, h), (245, 247, 252))
    if kind == "dark":
        return Image.new("RGB", (w, h), (10, 16, 28))
    top, bottom = (116, 91, 255), (15, 20, 35)
    img = Image.new("RGB", (w, h), top)
    px = img.load()
    for y in range(h):
        a = y / max(1, h - 1)
        r = int(top[0] * (1 - a) + bottom[0] * a)
        g = int(top[1] * (1 - a) + bottom[1] * a)
        b = int(top[2] * (1 - a) + bottom[2] * a)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img

def _draw_text_centered(base: Image.Image, text: str, style: str, anim: str, frame_i: int, frames: int) -> Image.Image:
    w, h = base.size
    font_size = 72 if style == "impactante" else 60 if style == "minimal" else 64
    p = frame_i / max(1, frames - 1)
    if anim == "fade":
        alpha, y_offset, scale = int(255 * min(1.0, max(0.0, p * 1.2))), 0, 1.0
    elif anim == "slide":
        alpha, y_offset, scale = 255, int((1 - p) * 80), 1.0
    else:
        alpha, y_offset, scale = 255, 0, 0.92 + (p * 0.10)

    canvas = base.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font_for_style(style, int(font_size * scale))
    padding_x = 64
    max_width = w - (padding_x * 2)
    lines = _wrap_text(draw, text, font, max_width)

    line_heights, line_widths = [], []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_h = sum(line_heights) + (len(lines) - 1) * 14
    start_y = (h - total_h) // 2 - y_offset
    fill, shadow = (255, 255, 255, alpha), (0, 0, 0, int(alpha * 0.65))

    y = start_y
    for i, line in enumerate(lines):
        x = (w - line_widths[i]) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += line_heights[i] + 14

    return Image.alpha_composite(canvas, overlay).convert("RGB")

def make_reels_video_styled(text: str, duration: int, out_path: Path, style="inspirador", bg="gradient", text_anim="fade", emoji_mode="auto"):
    w, h, fps = 720, 1280, 24
    frames = max(1, duration * fps)
    if emoji_mode == "auto":
        text = _auto_emojis(text)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for i in range(frames):
        base = _bg_image(w, h, bg)
        glow = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse((-200, -240, w + 200, 240), fill=(255, 255, 255, 38))
        base = Image.alpha_composite(base.convert("RGBA"), glow).convert("RGB")
        frame = _draw_text_centered(base, text, style, text_anim, i, frames)
        writer.append_data(np.array(frame))
    writer.close()

def choose_music_path(kind: str) -> Optional[Path]:
    if kind == "soft" and MUSIC_SOFT.exists():
        return MUSIC_SOFT
    if kind == "cinematic" and MUSIC_CINEMATIC.exists():
        return MUSIC_CINEMATIC
    if kind == "happy" and MUSIC_HAPPY.exists():
        return MUSIC_HAPPY
    return None

async def openai_tts_to_file(text: str, voice: str, out_path: Path, speed=1.0, response_format="mp3", instructions=None):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")
    payload = {"model": OPENAI_TTS_MODEL, "input": text[:4096], "voice": voice, "speed": speed, "response_format": response_format}
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
    return "onyx" if voice_gender == "male" else "nova"

def choose_voice_pair(language: str, voice_gender: str) -> tuple[str, str]:
    return ("onyx", "echo") if voice_gender == "male" else ("nova", "shimmer")

def concat_audios(a1: Path, a2: Path, out_mp3: Path) -> None:
    run_ffmpeg([FFMPEG_PATH, "-y", "-i", str(a1), "-i", str(a2), "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]", "-map", "[a]", "-c:a", "libmp3lame", "-q:a", "3", str(out_mp3)])

def mux_video_audio(video_path: Path, voice_path: Optional[Path], music_path: Optional[Path], out_path: Path):
    if voice_path and music_path:
        run_ffmpeg([FFMPEG_PATH, "-y", "-i", str(video_path), "-i", str(voice_path), "-stream_loop", "-1", "-i", str(music_path),
                    "-filter_complex", "[2:a]volume=0.25[m];[m][1:a]sidechaincompress=threshold=0.03:ratio=10:attack=20:release=250[mduck];[1:a][mduck]amix=inputs=2:duration=shortest[aout]",
                    "-map", "0:v:0", "-map", "[aout]", "-shortest", "-c:v", "copy", "-c:a", "aac", str(out_path)])
        return
    if voice_path and not music_path:
        run_ffmpeg([FFMPEG_PATH, "-y", "-i", str(video_path), "-i", str(voice_path), "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "copy", "-c:a", "aac", str(out_path)])
        return
    if (not voice_path) and music_path:
        run_ffmpeg([FFMPEG_PATH, "-y", "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path), "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "copy", "-c:a", "aac", str(out_path)])
        return
    out_path.write_bytes(video_path.read_bytes())

async def download_to_file(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.get(url)
        r.raise_for_status()
        out_path.write_bytes(r.content)

def build_tts_parts(text: str, language: str, mode: str) -> tuple[str, str, str, str]:
    if language == "es":
        narr_instr = "Habla en español latino neutro. Voz cálida, clara, estilo narrador."
        char_instr = "Español latino neutro. Más expresivo, como personaje de historia."
        char_text = "¡Wow! Dios sigue al control. Estoy contigo." if mode == "narrator_plus_character" else ""
        return text, narr_instr, char_text, char_instr
    narr_instr = "Clear English. Warm storyteller narrator style."
    char_instr = "English. More expressive, character style."
    char_text = "Wow! God is still in control. I’m with you." if mode == "narrator_plus_character" else ""
    return text, narr_instr, char_text, char_instr

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
            json={"model": OPENAI_MODEL, "messages": [{"role": "system", "content": "Eres amable, empático y claro."}, {"role": "user", "content": req.message}]},
        )
        r.raise_for_status()
        data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMAGE
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
    b64 = data["data"][0].get("b64_json") if data.get("data") else None
    if not b64:
        return {"status": "ok", "raw": data}
    img_bytes = base64.b64decode(b64)
    name = safe_filename("png")
    file_path(name).write_bytes(img_bytes)
    return {"status": "ok", "image_url": absolute_url(request, f"/files/{name}")}

# =========================
# REELS / TTS / REELS-VOICE
# =========================
@app.post("/reels")
def reels(req: VideoRequest, request: Request):
    base_name = f"reels_{uuid.uuid4().hex}.mp4"
    base_path = file_path(base_name)
    make_reels_video_styled(req.text, req.duration, base_path, req.style, req.bg, req.text_anim, req.emoji_mode)
    music_path = choose_music_path(req.music)
    if music_path:
        out_name = f"reels_music_{uuid.uuid4().hex}.mp4"
        out_path = file_path(out_name)
        mux_video_audio(base_path, None, music_path, out_path)
        return {"status": "ok", "video_url": absolute_url(request, f"/files/{out_name}")}
    return {"status": "ok", "video_url": absolute_url(request, f"/files/{base_name}")}

@app.post("/tts")
async def tts(req: TTSRequest, request: Request):
    name = safe_filename(req.response_format)
    out_path = file_path(name)
    await openai_tts_to_file(req.text, req.voice, out_path, req.speed, req.response_format, req.instructions)
    return {"status": "ok", "audio_url": absolute_url(request, f"/files/{name}")}

@app.post("/reels-voice")
async def reels_voice(req: ReelsVoiceRequest, request: Request):
    base_name = f"reels_{uuid.uuid4().hex}.mp4"
    base_path = file_path(base_name)
    make_reels_video_styled(req.text, req.duration, base_path, req.style, req.bg, req.text_anim, req.emoji_mode)
    if req.mode == "narrator_plus_character":
        v_narr, v_char = choose_voice_pair(req.language, req.voice_gender)
        narr_text, narr_instr, char_text, char_instr = build_tts_parts(req.text, req.language, req.mode)
        a1, a2 = file_path(safe_filename("mp3")), file_path(safe_filename("mp3"))
        await openai_tts_to_file(narr_text, v_narr, a1, instructions=narr_instr)
        await openai_tts_to_file(char_text, v_char, a2, instructions=char_instr)
        voice_path = file_path(safe_filename("mp3"))
        concat_audios(a1, a2, voice_path)
    else:
        voice_path = file_path(safe_filename("mp3"))
        instr = "Voz clara, cálida, ritmo natural." if req.language == "es" else "Clear, warm voice, natural pacing."
        await openai_tts_to_file(req.text, choose_voice(req.language, req.voice_gender), voice_path, instructions=instr)
    music_path = choose_music_path(req.music)
    out_name = f"reels_voice_{uuid.uuid4().hex}.mp4"
    out_path = file_path(out_name)
    mux_video_audio(base_path, voice_path, music_path, out_path)
    return {"status": "ok", "video_url": absolute_url(request, f"/files/{out_name}")}

# =========================
# VIDEO CINE (FAL)
# =========================
@app.post("/video-cine")
def video_cine(req: VideoRequest, request: Request):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}
    handler = fal_client.submit(VIDEO_MODEL_CINE, arguments={"prompt": req.text, "duration": req.duration})
    result = handler.get()
    video_url = (result.get("video", {}) or {}).get("url") or (result.get("videos", [{}])[0] or {}).get("url")
    if not video_url:
        return {"status": "error", "message": "No se obtuvo video_url", "raw": result}
    return {"status": "ok", "video_url": video_url}

# =========================
# VIDEO CINE + VOZ + MUSICA  (MEJORA MINIMA)
# =========================
@app.post("/video-cine-voice")
async def video_cine_voice(req: CineVoiceRequest, request: Request):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada"}
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no configurada"}

    handler = fal_client.submit(VIDEO_MODEL_CINE, arguments={"prompt": req.text, "duration": req.duration})
    result = handler.get()
    src_video_url = (result.get("video", {}) or {}).get("url") or (result.get("videos", [{}])[0] or {}).get("url")
    if not src_video_url:
        return {"status": "error", "message": "No se obtuvo video_url", "raw": result}

    base_name = f"cine_{uuid.uuid4().hex}.mp4"
    base_path = file_path(base_name)
    await download_to_file(src_video_url, base_path)

    if req.mode == "narrator_plus_character":
        v_narr, v_char = choose_voice_pair(req.language, req.voice_gender)
        narr_text, narr_instr, char_text, char_instr = build_tts_parts(req.text, req.language, req.mode)
        a1, a2 = file_path(safe_filename("mp3")), file_path(safe_filename("mp3"))
        await openai_tts_to_file(narr_text, v_narr, a1, instructions=narr_instr)
        await openai_tts_to_file(char_text, v_char, a2, instructions=char_instr)
        voice_path = file_path(safe_filename("mp3"))
        concat_audios(a1, a2, voice_path)
    else:
        voice_path = file_path(safe_filename("mp3"))
        instr = "Habla en español latino neutro. Voz cálida y natural." if req.language == "es" else "Clear English, warm and natural pacing."
        await openai_tts_to_file(req.text, choose_voice(req.language, req.voice_gender), voice_path, instructions=instr)

    music_path = choose_music_path(req.music)
    out_name = f"cine_voice_{uuid.uuid4().hex}.mp4"
    out_path = file_path(out_name)
    mux_video_audio(base_path, voice_path, music_path, out_path)

    final_url = absolute_url(request, f"/files/{out_name}")
    print("✅ /video-cine-voice OK:", {"video_url": final_url, "filename": out_name, "src_video_url": src_video_url})

    return {"status": "ok", "video_url": final_url, "filename": out_name, "src_video_url": src_video_url}
