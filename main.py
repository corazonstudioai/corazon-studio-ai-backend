import os
import uuid
from pathlib import Path
import base64

from fastapi import FastAPI
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

# FAL: llave desde Render -> Environment Variables
fal_client.api_key = os.getenv("FAL_KEY")

# Carpeta temporal para outputs (Render permite /tmp)
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
    text: str
    duration: int = 5
    format: str = "9:16"   # "9:16" o "16:9"
    cfg: float = 0.5       # (por ahora no lo usamos en Kling, se deja para UI)

# =========================
# HELPERS
# =========================
def extract_video_url(result: dict):
    """Intenta extraer una URL de video de respuestas comunes de FAL."""
    if not isinstance(result, dict):
        return None
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    vids = result.get("videos")
    if isinstance(vids, list) and len(vids) > 0 and isinstance(vids[0], dict) and vids[0].get("url"):
        return vids[0]["url"]
    out = result.get("output")
    if isinstance(out, dict) and out.get("url"):
        return out["url"]
    if result.get("url"):
        return result.get("url")
    return None


def extract_image_from_openai_images(resp_json: dict):
    """
    OpenAI images puede devolver:
      - data[0].url (si se pidió url)
      - data[0].b64_json (si devuelve base64)
    Devuelve:
      ("url", url) o ("b64", b64_string) o (None, None)
    """
    try:
        item = resp_json.get("data", [])[0]
        if item.get("url"):
            return ("url", item["url"])
        if item.get("b64_json"):
            return ("b64", item["b64_json"])
    except Exception:
        pass
    return (None, None)


async def make_openai_image_bytes(prompt: str) -> bytes:
    """
    Genera una imagen con OpenAI y devuelve bytes PNG/JPG.
    Preferimos b64_json (más confiable para luego pasar a FAL).
    Si solo llega url, la descargamos y regresamos los bytes.
    """
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": prompt,
                "size": "1024x1024",
                # Nota: no forzamos response_format porque puede variar,
                # pero nuestro extractor maneja url o b64_json.
            },
        )

    data = r.json()
    kind, val = extract_image_from_openai_images(data)

    if kind == "b64" and val:
        return base64.b64decode(val)

    if kind == "url" and val:
        async with httpx.AsyncClient(timeout=180) as client:
            img_r = await client.get(val)
            img_r.raise_for_status()
            return img_r.content

    raise RuntimeError("No se pudo obtener image_url/b64_json desde OpenAI")


def save_temp_image(image_bytes: bytes) -> Path:
    """Guarda bytes de imagen en /tmp para pasarlo a FAL."""
    img_path = OUT_DIR / f"tmp_{uuid.uuid4().hex}.png"
    img_path.write_bytes(image_bytes)
    return img_path


# =========================
# ROOT / HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}


# =========================
# UI BONITA (INCRUSTADA)
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
    .title{font-size:30px;font-weight:900;margin:10px 0 6px}
    .sub{opacity:.85;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:1fr;gap:14px}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px}
    textarea,input,select{width:100%;border-radius:12px;border:1px solid rgba(255,255,255,.16);background:rgba(0,0,0,.25);color:#fff;padding:12px;outline:none}
    button{width:100%;padding:12px 14px;border-radius:12px;border:0;background:linear-gradient(90deg,#2563eb,#7c3aed);color:white;font-weight:900;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .row{display:flex;gap:10px}
    .row>*{flex:1}
    .out{white-space:pre-wrap;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px;min-height:68px}
    .small{font-size:12px;opacity:.85;margin-top:8px}
    img,video{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.12);margin-top:10px}
    .hint{opacity:.8;font-size:12px;margin-top:8px}
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
      <textarea id="vidText" rows="3" placeholder="Ej: Una niña sonriendo caminando en un parque al atardecer, estilo cinematográfico, luz cálida, cámara suave, realista"></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <select id="vidDur">
          <option value="3">3s</option>
          <option value="5" selected>5s</option>
        </select>
        <select id="vidFmt">
          <option value="9:16" selected>9:16</option>
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
      const url = data?.data?.[0]?.url;
      if(url){
        img.src = url;
        img.style.display="block";
        info.textContent = "✅ imagen lista";
      }else{
        info.textContent = "✅ respuesta recibida (sin url directa).";
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
      out.textContent = "Listo ✅ Descargando reels.mp4";
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(reelsBtn,false);
  };

  // VIDEO CINE (texto->imagen->video)
  const vidBtn = document.getElementById("vidBtn");
  vidBtn.dataset.label = "Generar Video";
  vidBtn.onclick = async () => {
    setLoading(vidBtn,true);
    const text = document.getElementById("vidText").value.trim();
    const duration = parseInt(document.getElementById("vidDur").value || "5",10);
    const format = document.getElementById("vidFmt").value;
    const out = document.getElementById("vidOut");
    const vid = document.getElementById("vidPreview");
    out.textContent = "";
    vid.style.display="none";
    try{
      const r = await fetch(apiBase + "/video-cine", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format, cfg: 0.5})
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
                    {"role": "system", "content": "Eres amable, empático y claro."},
                    {"role": "user", "content": req.message},
                ],
            },
        )
    return {"reply": r.json()["choices"][0]["message"]["content"]}


# =========================
# IMAGE
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
# REELS MP4 (LOCAL)
# =========================
def make_reels(text: str, duration: int, fmt: str, out_path: Path):
    w, h = (1280, 720) if (fmt or "9:16").strip() == "16:9" else (720, 1280)
    fps = 24
    frames = max(1, duration) * fps

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    # Simple: texto centrado
    for i in range(frames):
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)

        tw = draw.textlength(text, font=font) if text else 0
        tx = max(20, (w - tw) / 2)
        ty = h // 2

        draw.text((tx, ty), text or " ", font=font, fill=(255, 255, 255))
        writer.append_data(np.array(img))

    writer.close()


@app.post("/reels")
def reels(req: VideoRequest):
    name = f"reels_{uuid.uuid4().hex}.mp4"
    path = OUT_DIR / name
    make_reels(req.text or " ", int(req.duration), req.format, path)
    return FileResponse(path, media_type="video/mp4", filename=name)


# =========================
# 🎬 VIDEO CINE REALISTA (TEXTO → IMAGEN → VIDEO)
# =========================
@app.post("/video-cine")
async def video_cine(req: VideoRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no configurada en Render"}
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no configurada en Render"}

    try:
        # 1) Texto -> Imagen (OpenAI)
        img_bytes = await make_openai_image_bytes(req.text)

        # 2) Guardar temporalmente en /tmp
        img_path = save_temp_image(img_bytes)

        # 3) Imagen -> Video (FAL Kling v2.6)
        aspect = "9:16" if (req.format or "9:16") == "9:16" else "16:9"

        result = fal_client.run(
            "fal-ai/kling-video/v2.6",
            arguments={
                "image_url": str(img_path),  # fal_client acepta path local en muchos flujos
                "prompt": req.text,
                "duration": max(1, int(req.duration)),
                "aspect_ratio": aspect,
            },
        )

        video_url = extract_video_url(result)
        if not video_url:
            return {"status": "error", "message": "No se obtuvo video_url", "raw": result}

        # (opcional) limpiar imagen temporal
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass

        return {"status": "ok", "video_url": video_url}

    except Exception as e:
        return {"status": "error", "message": "No se pudo generar video", "detail": str(e)}
```0
