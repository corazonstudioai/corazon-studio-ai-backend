import os
import uuid
import base64
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
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

# Puedes dejar estas por si luego las usas
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
PIKA_API_KEY = os.getenv("PIKA_API_KEY")

# ✅ FAL key (Render Environment Variable)
fal_client.api_key = os.getenv("FAL_KEY")

# Carpeta temporal para guardar archivos
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
    mode: str  # reels | cine | runway | pika
    text: str = ""
    duration: int = 4
    format: str = "9:16"  # "9:16" o "16:9"

# =========================
# HELPERS
# =========================
def _base_url(req: Request) -> str:
    # Devuelve el dominio correcto para construir urls como /file/xxx.png
    # Ej: https://corazon-studio-ai-backend-3.onrender.com
    return str(req.base_url).rstrip("/")

def _save_b64_png_to_tmp(b64_data: str) -> Path:
    """
    Convierte b64_json (OpenAI) a PNG en /tmp/out.
    Retorna el path del archivo guardado.
    """
    out_name = f"img_{uuid.uuid4().hex}.png"
    out_path = OUT_DIR / out_name
    img_bytes = base64.b64decode(b64_data)
    out_path.write_bytes(img_bytes)
    return out_path

def _extract_video_url(result):
    """
    Intenta sacar una URL de video desde respuestas comunes de FAL.
    """
    if not isinstance(result, dict):
        return None

    # casos típicos
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v.get("url")

    vids = result.get("videos")
    if isinstance(vids, list) and len(vids) > 0:
        first = vids[0]
        if isinstance(first, dict) and first.get("url"):
            return first.get("url")

    out = result.get("output")
    if isinstance(out, dict) and out.get("url"):
        return out.get("url")

    if result.get("url"):
        return result.get("url")

    return None

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

# =========================
# SERVIR ARCHIVOS TEMPORALES (IMÁGENES/VIDEOS)
# =========================
@app.get("/file/{name}")
def get_file(name: str):
    # Seguridad básica: solo nombres de archivo, sin rutas
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

    path = OUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Detectar tipo básico por extensión
    ext = path.suffix.lower()
    media = "application/octet-stream"
    if ext in [".png"]:
        media = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        media = "image/jpeg"
    elif ext in [".mp4"]:
        media = "video/mp4"

    return FileResponse(str(path), media_type=media, filename=name)

# =========================
# UI BONITA (BOTONES)
# =========================
@app.get("/app", response_class=HTMLResponse)
def app_ui():
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Corazón Studio AI</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:#0b1020;color:#e9eeff}
    .wrap{max-width:980px;margin:0 auto;padding:22px}
    .title{font-size:28px;font-weight:900;margin:10px 0 6px}
    .sub{opacity:.85;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:1fr;gap:14px}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px}
    textarea,input,select{width:100%;border-radius:12px;border:1px solid rgba(255,255,255,.16);background:rgba(0,0,0,.25);color:#fff;padding:12px;outline:none}
    button{width:100%;padding:12px 14px;border-radius:12px;border:0;background:#7c3aed;color:white;font-weight:900;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .row{display:flex;gap:10px}
    .row>*{flex:1}
    .out{white-space:pre-wrap;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px;min-height:68px}
    .small{font-size:12px;opacity:.85;margin-top:8px}
    img,video{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.12);margin-top:10px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Corazón Studio AI ✨</div>
  <p class="sub">Chat, Imagen, Reels MP4 y Video cine realista (texto → video).</p>

  <div class="grid">
    <div class="card">
      <h3>💬 Chat</h3>
      <textarea id="chatText" rows="3" placeholder="Escribe tu mensaje..."></textarea>
      <div style="height:10px"></div>
      <button id="chatBtn">Enviar</button>
      <div style="height:10px"></div>
      <div class="out" id="chatOut"></div>
      <div class="small">Tip: si no responde, revisa que el backend esté activo.</div>
    </div>

    <div class="card">
      <h3>🖼️ Imagen (OpenAI)</h3>
      <textarea id="imgText" rows="3" placeholder="Describe la imagen..."></textarea>
      <div style="height:10px"></div>
      <button id="imgBtn">Generar</button>
      <div id="imgOut" class="small"></div>
      <img id="imgPreview" style="display:none" />
    </div>

    <div class="card">
      <h3>🎞️ Reels MP4 (local)</h3>
      <textarea id="reelsText" rows="3" placeholder="Texto para el reels..."></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <input id="reelsDur" type="number" value="6" min="1" />
        <select id="reelsFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>
      <div style="height:10px"></div>
      <button id="reelsBtn">Crear Reels (descargar MP4)</button>
      <div id="reelsOut" class="small"></div>
    </div>

    <div class="card">
      <h3>🎬 Video cine realista (texto → video)</h3>
      <textarea id="vidText" rows="3" placeholder="Ej: Una niña feliz caminando en un parque al atardecer, estilo cine realista..."></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <input id="vidDur" type="number" value="4" min="1" max="10" />
        <select id="vidFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>
      <div style="height:10px"></div>
      <button id="vidBtn">Generar Video</button>
      <div id="vidOut" class="small"></div>
      <video id="vidPreview" controls playsinline style="display:none"></video>
    </div>
  </div>
</div>

<script>
  const apiBase = "";

  function setLoading(btn, on){
    btn.disabled = on;
    btn.textContent = on ? "Procesando..." : btn.dataset.label;
  }

  // CHAT
  const chatBtn = document.getElementById("chatBtn");
  chatBtn.dataset.label = "Enviar";
  chatBtn.onclick = async () => {
    setLoading(chatBtn,true);
    const message = document.getElementById("chatText").value.trim();
    const out = document.getElementById("chatOut");
    out.textContent = "";
    try{
      const r = await fetch(apiBase + "/chat", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({message})
      });
      const data = await r.json();
      out.textContent = data.reply ?? JSON.stringify(data,null,2);
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(chatBtn,false);
  };

  // IMAGEN
  const imgBtn = document.getElementById("imgBtn");
  imgBtn.dataset.label = "Generar";
  imgBtn.onclick = async () => {
    setLoading(imgBtn,true);
    const prompt = document.getElementById("imgText").value.trim();
    const info = document.getElementById("imgOut");
    const img = document.getElementById("imgPreview");
    info.textContent = "";
    img.style.display="none";
    try{
      const r = await fetch(apiBase + "/image", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({prompt})
      });
      const data = await r.json();
      if(data.image_url){
        img.src = data.image_url;
        img.style.display="block";
        info.textContent = "✅ Imagen lista";
      }else{
        info.textContent = "Respuesta: " + JSON.stringify(data,null,2);
      }
    }catch(e){
      info.textContent = "Error: " + e;
    }
    setLoading(imgBtn,false);
  };

  // REELS
  const reelsBtn = document.getElementById("reelsBtn");
  reelsBtn.dataset.label = "Crear Reels (descargar MP4)";
  reelsBtn.onclick = async () => {
    setLoading(reelsBtn,true);
    const text = document.getElementById("reelsText").value.trim();
    const duration = parseInt(document.getElementById("reelsDur").value || "6",10);
    const format = document.getElementById("reelsFmt").value;
    const out = document.getElementById("reelsOut");
    out.textContent = "";
    try{
      const r = await fetch(apiBase + "/video", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({mode:"reels", text, duration, format})
      });
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "reels.mp4";
      a.click();
      out.textContent = "✅ Listo. Descargando reels.mp4";
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(reelsBtn,false);
  };

  // VIDEO CINE
  const vidBtn = document.getElementById("vidBtn");
  vidBtn.dataset.label = "Generar Video";
  vidBtn.onclick = async () => {
    setLoading(vidBtn,true);
    const text = document.getElementById("vidText").value.trim();
    const duration = parseInt(document.getElementById("vidDur").value || "4",10);
    const format = document.getElementById("vidFmt").value;
    const out = document.getElementById("vidOut");
    const vid = document.getElementById("vidPreview");
    out.textContent = "";
    vid.style.display="none";
    try{
      const r = await fetch(apiBase + "/video", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({mode:"cine", text, duration, format})
      });
      const data = await r.json();
      if(data.video_url){
        vid.src = data.video_url;
        vid.style.display="block";
        out.textContent = "✅ Video listo";
      }else{
        out.textContent = "Respuesta: " + JSON.stringify(data,null,2);
      }
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(vidBtn,false);
  };
</script>
</body>
</html>
"""

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
                    {"role": "system", "content": "Eres un asistente amable, profesional y empático."},
                    {"role": "user", "content": req.message},
                ],
            },
        )
    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMÁGENES (ARREGLADO: URL o b64_json → image_url SIEMPRE)
# =========================
@app.post("/image")
async def image(req: ImageRequest, request: Request):
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

    # Caso A: OpenAI devuelve url directa
    url = data.get("data", [{}])[0].get("url")
    if url:
        return {"status": "ok", "image_url": url}

    # Caso B: OpenAI devuelve b64_json
    b64 = data.get("data", [{}])[0].get("b64_json")
    if b64:
        saved = _save_b64_png_to_tmp(b64)
        img_url = f"{_base_url(request)}/file/{saved.name}"
        return {"status": "ok", "image_url": img_url}

    # Si no vino nada usable:
    return {"status": "error", "message": "No se pudo obtener imagen desde OpenAI", "raw": data}

# =========================
# HELPERS: REELS MP4
# =========================
def _reels_size(fmt: str):
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

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    lines = _wrap_text(text, max_chars=22)

    for i in range(total_frames):
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)

        t = i / max(1, total_frames - 1)
        y_base = int(h * (0.62 - 0.08 * (1 - np.cos(np.pi * t)) / 2))

        box_margin = 60
        box_w = w - 2 * box_margin
        box_h = 70 * len(lines) + 80
        x0 = box_margin
        y0 = y_base - box_h // 2
        x1 = x0 + box_w
        y1 = y0 + box_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(18, 27, 46))

        y = y0 + 40
        for ln in lines:
            tw = draw.textlength(ln, font=font)
            tx = (w - tw) / 2
            draw.text((tx, y), ln, font=font, fill=(255, 255, 255))
            y += 70

        writer.append_data(np.array(img))

    writer.close()

# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest, request: Request):
    mode = (req.mode or "").strip().lower()

    # -------- MODO REELS (MP4 descargable)
    if mode == "reels":
        out_name = f"reels_{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name
        _make_reels_mp4(req.text or " ", max(1, int(req.duration)), req.format, out_path)
        return FileResponse(path=str(out_path), media_type="video/mp4", filename=out_name)

    # -------- MODO CINE (texto → video realista, estable)
    # Usamos Runway Gen2 via FAL (NO depende de image_url, así se evita el error)
    if mode in ["cine", "runway"]:
        if not fal_client.api_key:
            return {"status": "error", "message": "FAL_KEY no está configurada en Render"}

        try:
            prompt = (req.text or "").strip()
            if not prompt:
                return {"status": "error", "message": "Escribe un prompt para el video"}

            # Prompt cinematográfico extra (para que salga mejor)
            cinematic = (
                f"{prompt}. Estilo cine realista, iluminación cinematográfica, "
                "movimiento natural de cámara, detalle alto, colores naturales."
            )

            result = fal_client.run(
                "runwayml/gen2",
                arguments={
                    "prompt": cinematic,
                    "seconds": max(1, int(req.duration)),
                },
            )

            video_url = _extract_video_url(result)
            if not video_url:
                return {"status": "error", "message": "No llegó video_url", "raw": result}

            return {"status": "ok", "provider": "runway", "video_url": video_url, "result": result}

        except Exception as e:
            return {"status": "error", "message": "FAL error", "detail": str(e)}

    # -------- MODO PIKA (si lo quieres)
    if mode == "pika":
        if not fal_client.api_key:
            return {"status": "error", "message": "FAL_KEY no está configurada en Render"}
        try:
            result = fal_client.run(
                "pika/video",
                arguments={
                    "prompt": req.text,
                    "seconds": max(1, int(req.duration)),
                },
            )
            video_url = _extract_video_url(result)
            return {"status": "ok", "provider": "pika", "video_url": video_url, "result": result}
        except Exception as e:
            return {"status": "error", "message": "FAL error", "detail": str(e)}

    return {"status": "error", "message": "Modo de video no válido. Usa: reels | cine | runway | pika"}
