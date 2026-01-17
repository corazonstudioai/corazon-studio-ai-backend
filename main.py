import os
import uuid
import base64
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
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

# FAL (Render env var: FAL_KEY)
FAL_KEY = os.getenv("FAL_KEY")
fal_client.api_key = FAL_KEY

# Output dir (Render ok)
OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# APP
# =========================
app = FastAPI(title="Corazón Studio AI Backend")

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
    size: str = "1024x1024"  # "1024x1024" o "512x512"

class ReelsRequest(BaseModel):
    text: str
    duration: int = 6
    format: Literal["9:16", "16:9"] = "9:16"

class CineVideoRequest(BaseModel):
    text: str = Field(..., description="Prompt de video")
    duration: int = Field(5, ge=1, le=10)
    format: Literal["9:16", "16:9"] = "9:16"
    # estos campos los aceptamos aunque Runway Gen2 no los use siempre (no rompen nada)
    cfg: float = Field(0.5, ge=0.0, le=10.0)
    quality: Literal["512p", "720p"] = "512p"

# =========================
# HELPERS
# =========================
def _extract_video_url(result):
    """
    Extrae URL de video de respuestas típicas de FAL.
    """
    if not isinstance(result, dict):
        return None

    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]

    vids = result.get("videos")
    if isinstance(vids, list) and len(vids) > 0:
        first = vids[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]

    out = result.get("output")
    if isinstance(out, dict) and out.get("url"):
        return out["url"]

    if result.get("url"):
        return result["url"]

    return None


def _reels_size(fmt: str):
    return (1280, 720) if fmt.strip() == "16:9" else (720, 1280)


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


def _image_api_to_url_or_datauri(openai_json: dict) -> dict:
    """
    OpenAI Images API puede devolver:
    - data[0].url
    - data[0].b64_json

    Devolvemos:
      { "type": "url", "value": "https://..." }
    o { "type": "datauri", "value": "data:image/png;base64,..." }
    """
    try:
        item = openai_json.get("data", [None])[0] or {}
        url = item.get("url")
        if url:
            return {"type": "url", "value": url}

        b64 = item.get("b64_json")
        if b64:
            return {"type": "datauri", "value": f"data:image/png;base64,{b64}"}
    except Exception:
        pass

    return {"type": "raw", "value": openai_json}

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

@app.get("/app", response_class=HTMLResponse)
def app_ui():
    # UI simple y bonita para probar desde el navegador (mismo backend).
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
    .title{font-size:30px;font-weight:900;margin:10px 0 6px}
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
    .small{font-size:12px;opacity:.8;margin-top:8px}
    img,video{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.12);margin-top:10px}
    .hint{font-size:12px;opacity:.8;margin-top:8px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Corazón Studio AI ✨</div>
  <p class="sub">Chat, imágenes, reels MP4 local y video cine realista (texto → video).</p>

  <div class="grid">
    <div class="card">
      <h3>💬 Chat</h3>
      <textarea id="chatText" rows="3" placeholder="Escribe tu mensaje..."></textarea>
      <div style="height:10px"></div>
      <button id="chatBtn">Enviar</button>
      <div class="hint">Tip: si no responde, revisa que el backend esté activo.</div>
      <div style="height:10px"></div>
      <div class="out" id="chatOut"></div>
    </div>

    <div class="card">
      <h3>🖼️ Imagen</h3>
      <textarea id="imgText" rows="3" placeholder="Describe la imagen..."></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <select id="imgSize">
          <option value="1024x1024">1024x1024</option>
          <option value="512x512">512x512</option>
        </select>
        <button id="imgBtn">Generar</button>
      </div>
      <div id="imgOut" class="small"></div>
      <img id="imgPreview" style="display:none" />
    </div>

    <div class="card">
      <h3>🎞️ Reels MP4 (local)</h3>
      <textarea id="reelsText" rows="3" placeholder="Texto para el reels..."></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <input id="reelsDur" type="number" value="6" min="1" max="20" />
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
      <textarea id="vidText" rows="4" placeholder="Ej: Una niña sonriendo caminando en un parque al atardecer, estilo cinematográfico, luz cálida, cámara suave, realista"></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <select id="vidDur">
          <option value="3">3s</option>
          <option value="5" selected>5s</option>
          <option value="8">8s</option>
          <option value="10">10s</option>
        </select>
        <select id="vidFmt">
          <option value="9:16" selected>9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>
      <div style="height:10px"></div>
      <div class="row">
        <input id="vidCfg" type="number" step="0.1" value="0.5" min="0" max="10" />
        <select id="vidQ">
          <option value="512p" selected>Calidad 512p</option>
          <option value="720p">Calidad 720p</option>
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

  // IMAGE
  const imgBtn = document.getElementById("imgBtn");
  imgBtn.dataset.label = "Generar";
  imgBtn.onclick = async () => {
    setLoading(imgBtn,true);
    const prompt = document.getElementById("imgText").value.trim();
    const size = document.getElementById("imgSize").value;
    const info = document.getElementById("imgOut");
    const img = document.getElementById("imgPreview");
    info.textContent = "";
    img.style.display="none";
    try{
      const r = await fetch(apiBase + "/image", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({prompt, size})
      });
      const data = await r.json();
      if(data.type === "url" || data.type === "datauri"){
        img.src = data.value;
        img.style.display="block";
        info.textContent = "✅ Imagen lista";
      }else{
        info.textContent = "Respuesta recibida (sin url directa): " + JSON.stringify(data,null,2);
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
      const r = await fetch(apiBase + "/reels", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format})
      });
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "reels.mp4";
      a.click();
      out.textContent = "✅ Descargando reels.mp4";
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
    const duration = parseInt(document.getElementById("vidDur").value || "5",10);
    const format = document.getElementById("vidFmt").value;
    const cfg = parseFloat(document.getElementById("vidCfg").value || "0.5");
    const quality = document.getElementById("vidQ").value;

    const out = document.getElementById("vidOut");
    const vid = document.getElementById("vidPreview");
    out.textContent = "";
    vid.style.display="none";
    try{
      const r = await fetch(apiBase + "/video-cine", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format, cfg, quality})
      });
      const data = await r.json();
      if(data.status === "ok" && data.video_url){
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
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no está configurada en Render"}

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
    try:
        return {"reply": data["choices"][0]["message"]["content"]}
    except Exception:
        return {"status": "error", "raw": data}

# =========================
# IMAGE
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no está configurada en Render"}

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
                "size": req.size,
            },
        )

    data = r.json()
    return _image_api_to_url_or_datauri(data)

# =========================
# REELS (LOCAL MP4)
# =========================
@app.post("/reels")
def reels(req: ReelsRequest):
    out_name = f"reels_{uuid.uuid4().hex}.mp4"
    out_path = OUT_DIR / out_name

    _make_reels_mp4(req.text or " ", max(1, int(req.duration)), req.format, out_path)

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=out_name,
    )

# =========================
# 🎬 VIDEO CINE REALISTA (TEXTO → VIDEO)
# =========================
@app.post("/video-cine")
def video_cine(req: CineVideoRequest):
    if not FAL_KEY:
        return {"status": "error", "message": "FAL_KEY no está configurada en Render"}

    # Runway Gen2 vía FAL (estable). Evitamos Kling por 'Application not found'.
    try:
        # Algunos parámetros dependen del wrapper; mandamos lo esencial:
        # - prompt
        # - seconds
        # - (opcional) aspect ratio: si no lo soporta, simplemente lo ignora o falla.
        args = {
            "prompt": req.text,
            "seconds": int(req.duration),
        }

        # Aspect ratio (si el modelo/adapter lo soporta)
        # Algunos adapters usan "aspect_ratio" o "ratio". Probamos con "aspect_ratio" sin romper.
        if req.format == "9:16":
            args["aspect_ratio"] = "9:16"
        else:
            args["aspect_ratio"] = "16:9"

        # (cfg/quality) los dejamos como extras; si el adapter no los usa, no pasa nada.
        args["cfg"] = float(req.cfg)
        args["quality"] = req.quality

        result = fal_client.run("runwayml/gen2", arguments=args)
        video_url = _extract_video_url(result)

        if not video_url:
            return {"status": "error", "message": "No se obtuvo video_url", "raw": result}

        return {"status": "ok", "provider": "fal:runwayml/gen2", "video_url": video_url}

    except Exception as e:
        return {"status": "error", "message": "FAL error", "detail": str(e)}
