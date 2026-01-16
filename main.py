import os
import uuid
import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
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

# FAL (Render Environment Variable)
# OJO: tu llave de FAL NO tiene que empezar con "FAL". Solo debe estar en FAL_KEY.
fal_client.api_key = os.getenv("FAL_KEY")

# Carpeta temporal (Render permite /tmp)
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
    mode: str  # reels | fal | runway | pika (fal recomendado)
    text: str = ""
    duration: int = 5          # 5 o 10 para Kling
    format: str = "9:16"       # 9:16 | 16:9 | 1:1
    quality: str = "512p"      # 512p | 720p (solo etiqueta UI)
    cfg: float = 0.5           # 0..1 (Kling)

# =========================
# HELPERS
# =========================
def _extract_video_url(result: dict):
    """Extrae URL de video desde salidas típicas."""
    if not isinstance(result, dict):
        return None

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

def _save_bytes(tmp_bytes: bytes, suffix: str) -> str:
    """Guarda bytes en /tmp y devuelve la ruta."""
    name = f"{uuid.uuid4().hex}{suffix}"
    path = OUT_DIR / name
    path.write_bytes(tmp_bytes)
    return str(path)

def _openai_image_url_or_data(payload: dict):
    """
    OpenAI Images puede regresar:
    - data[0].url
    - data[0].b64_json
    Retornamos (kind, value) donde kind: "url" | "data_url" | None
    """
    try:
        item = payload.get("data", [None])[0]
        if not item:
            return None, None

        if item.get("url"):
            return "url", item["url"]

        b64 = item.get("b64_json")
        if b64:
            raw = base64.b64decode(b64)
            data_url = "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")
            return "data_url", data_url
    except Exception:
        pass
    return None, None

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

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
    :root{
      --bg:#070b18; --card:rgba(255,255,255,.06); --stroke:rgba(255,255,255,.12);
      --ink:#eaf0ff; --muted:rgba(234,240,255,.78);
      --btn1:#2563eb; --btn2:#7c3aed;
    }
    *{box-sizing:border-box}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:radial-gradient(1200px 600px at 30% -10%, rgba(124,58,237,.35), transparent 60%),
                                                     radial-gradient(900px 500px at 90% 10%, rgba(37,99,235,.25), transparent 55%),
                                                     var(--bg);
         color:var(--ink)}
    .wrap{max-width:980px;margin:0 auto;padding:22px}
    .title{font-size:30px;font-weight:900;margin:12px 0 4px;letter-spacing:.2px}
    .sub{margin:0 0 18px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:1fr;gap:14px}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:var(--card);border:1px solid var(--stroke);border-radius:18px;padding:16px;backdrop-filter: blur(10px)}
    h3{margin:0 0 10px}
    label{display:block;font-size:12px;color:var(--muted);margin:10px 0 6px}
    textarea,input,select{
      width:100%;border-radius:12px;border:1px solid rgba(255,255,255,.16);
      background:rgba(0,0,0,.25);color:#fff;padding:12px;outline:none
    }
    button{
      width:100%;padding:12px 14px;border-radius:12px;border:0;
      background:linear-gradient(90deg,var(--btn1),var(--btn2));
      color:white;font-weight:900;cursor:pointer
    }
    button:disabled{opacity:.6;cursor:not-allowed}
    .row{display:flex;gap:10px}
    .row>*{flex:1}
    .out{
      white-space:pre-wrap;background:rgba(0,0,0,.25);
      border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px;min-height:68px
    }
    .hint{font-size:12px;color:var(--muted);margin-top:8px}
    img,video{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.12);margin-top:10px}
    .ok{color:#a7f3d0}
    .err{color:#fecaca}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Corazón Studio AI ✨</div>
  <p class="sub">Chat, imágenes y <b>video cine realista (texto → video)</b>.</p>

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
      <button id="imgBtn">Generar</button>
      <div id="imgOut" class="hint"></div>
      <img id="imgPreview" style="display:none" />
    </div>

    <div class="card">
      <h3>🎞️ Reels MP4 (local)</h3>
      <textarea id="reelsText" rows="3" placeholder="Texto para el reels..."></textarea>
      <label>Duración y formato</label>
      <div class="row">
        <input id="reelsDur" type="number" value="6" min="1" />
        <select id="reelsFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>
      <div style="height:10px"></div>
      <button id="reelsBtn">Crear Reels (descargar MP4)</button>
      <div id="reelsOut" class="hint"></div>
    </div>

    <div class="card">
      <h3>🎬 Video cine realista (texto → video)</h3>
      <textarea id="vidText" rows="3" placeholder="Ej: Una niña sonriente caminando en un parque al atardecer, estilo cinematográfico, luz cálida, cámara suave, realista"></textarea>

      <label>Duración y formato</label>
      <div class="row">
        <select id="vidDur">
          <option value="5">5s</option>
          <option value="10">10s</option>
        </select>
        <select id="vidFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
          <option value="1:1">1:1</option>
        </select>
      </div>

      <label>CFG (qué tanto obedece el prompt)</label>
      <div class="row">
        <input id="vidCfg" type="number" value="0.5" min="0" max="1" step="0.1" />
        <select id="vidQuality">
          <option value="512p">Calidad 512p (más rápido)</option>
          <option value="720p">Calidad 720p (más lento)</option>
        </select>
      </div>

      <div style="height:10px"></div>
      <button id="vidBtn">Generar Video</button>
      <div id="vidOut" class="hint"></div>
      <video id="vidPreview" controls playsinline style="display:none"></video>
    </div>
  </div>
</div>

<script>
  const apiBase = ""; // mismo dominio del backend

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
      if(data?.url){
        img.src = data.url;
        img.style.display="block";
        info.innerHTML = '<span class="ok">✅ ¡imagen lista!</span>';
      }else if(data?.data_url){
        img.src = data.data_url;
        img.style.display="block";
        info.innerHTML = '<span class="ok">✅ ¡imagen lista!</span>';
      }else{
        info.innerHTML = '<span class="err">⚠️ No llegó url/data_url</span><br/>' + JSON.stringify(data,null,2);
      }
    }catch(e){
      info.innerHTML = '<span class="err">Error:</span> ' + e;
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
      out.innerHTML = '<span class="ok">Listo ✅ Descargando reels.mp4</span>';
    }catch(e){
      out.innerHTML = '<span class="err">Error:</span> ' + e;
    }
    setLoading(reelsBtn,false);
  };

  // VIDEO CINE (FAL -> Kling v2 master)
  const vidBtn = document.getElementById("vidBtn");
  vidBtn.dataset.label = "Generar Video";
  vidBtn.onclick = async () => {
    setLoading(vidBtn,true);
    const prompt = document.getElementById("vidText").value.trim();
    const duration = parseInt(document.getElementById("vidDur").value || "5", 10);
    const format = document.getElementById("vidFmt").value;
    const cfg = parseFloat(document.getElementById("vidCfg").value || "0.5");
    const quality = document.getElementById("vidQuality").value;

    const out = document.getElementById("vidOut");
    const vid = document.getElementById("vidPreview");
    out.textContent = "";
    vid.style.display="none";

    try{
      const r = await fetch(apiBase + "/video", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({mode:"fal", text: prompt, duration, format, quality, cfg})
      });
      const data = await r.json();
      if(data.video_url){
        vid.src = data.video_url;
        vid.style.display="block";
        out.innerHTML = '<span class="ok">Video listo ✅</span>';
      }else{
        out.innerHTML = '<span class="err">No llegó video_url</span><br/>' + JSON.stringify(data,null,2);
      }
    }catch(e){
      out.innerHTML = '<span class="err">Error:</span> ' + e;
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
        return {"status": "error", "message": "OPENAI_API_KEY no está configurada"}
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
# IMÁGENES (SIEMPRE DEVUELVE algo visible: url o data_url)
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no está configurada"}

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

    payload = r.json()
    kind, value = _openai_image_url_or_data(payload)
    if kind == "url":
        return {"status": "ok", "url": value}
    if kind == "data_url":
        return {"status": "ok", "data_url": value}

    # fallback: devuelve respuesta cruda para ver el error exacto
    return {"status": "error", "message": "No se pudo obtener url/b64_json", "raw": payload}

# =========================
# HELPERS: REELS MP4 (local)
# =========================
def _reels_size(fmt: str):
    return (1280, 720) if fmt.strip() == "16:9" else (720, 1280)

def _wrap_text(text: str, max_chars: int = 22):
    words = text.split()
    lines, line = [], ""
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
# - reels: genera mp4 local (descarga)
# - fal: texto->video realista con Kling v2 master (FAL)
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    mode = (req.mode or "").strip().lower()

    # -------- A) REELS MP4 LOCAL
    if mode == "reels":
        out_name = f"reels_{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name
        _make_reels_mp4(req.text or " ", max(1, int(req.duration)), req.format, out_path)
        return FileResponse(path=str(out_path), media_type="video/mp4", filename=out_name)

    # -------- B) VIDEO CINE REALISTA (TEXTO->VIDEO) CON FAL KLING V2 MASTER
    if mode == "fal":
        if not os.getenv("FAL_KEY"):
            return {"status": "error", "message": "FAL_KEY no está configurada en Render"}

        prompt = (req.text or "").strip()
        if not prompt:
            return {"status": "error", "message": "Escribe un prompt para el video"}

        # Kling master solo acepta "5" o "10" como string
        dur = "10" if int(req.duration) >= 10 else "5"

        # Kling master acepta: 16:9 | 9:16 | 1:1
        ar = (req.format or "9:16").strip()
        if ar not in ("16:9", "9:16", "1:1"):
            ar = "9:16"

        cfg = float(req.cfg) if req.cfg is not None else 0.5
        if cfg < 0: cfg = 0.0
        if cfg > 1: cfg = 1.0

        try:
            result = fal_client.subscribe(
                "fal-ai/kling-video/v2/master/text-to-video",
                arguments={
                    "prompt": prompt,
                    "duration": dur,          # "5" o "10"
                    "aspect_ratio": ar,       # "9:16" etc
                    "cfg_scale": cfg,         # 0..1
                    "negative_prompt": "blur, distort, low quality, text, subtitles, watermark",
                },
            )
            video_url = _extract_video_url(result)
            if not video_url:
                return {"status": "error", "message": "No llegó video_url", "raw": result}
            return {"status": "ok", "provider": "fal-kling-v2-master", "video_url": video_url}
        except Exception as e:
            return {"status": "error", "message": "FAL error", "detail": str(e)}

    # (Opcionales) si en tu UI te aparece runway/pika, aquí lo bloqueamos para evitar errores raros:
    if mode in ("runway", "pika"):
        return {
            "status": "error",
            "message": "Este modo está desactivado por ahora. Usa modo 'fal' (Kling) que es el estable.",
        }

    return {"status": "error", "message": "Modo de video no válido. Usa: reels | fal"}
