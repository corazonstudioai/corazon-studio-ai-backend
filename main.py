import os
import uuid
import base64
from pathlib import Path
from typing import Optional, Any, Dict

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

# FAL KEY (en Render debe llamarse EXACTO: FAL_KEY)
fal_client.api_key = os.getenv("FAL_KEY")

# Carpeta temporal
OUT_DIR = Path("/tmp/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelo FAL recomendado (EXISTE en docs)
FAL_KLING_IMAGE_TO_VIDEO = "fal-ai/kling-video/v2.1/standard/image-to-video"


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


class ReelsRequest(BaseModel):
    text: str = ""
    duration: int = 6
    format: str = "9:16"  # "9:16" o "16:9"


class VideoCineRequest(BaseModel):
    text: str
    duration: int = 5              # 5 o 10 (según modelo)
    format: str = "9:16"           # 9:16 o 16:9
    cfg: float = 0.5               # cfg_scale (qué tanto obedece)
    quality: str = "512p"          # 512p o 720p (según modelo)


# =========================
# HELPERS
# =========================
def _extract_video_url(result: Any) -> Optional[str]:
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
        return result["url"]

    return None


def _aspect_ratio_from_format(fmt: str) -> str:
    fmt = (fmt or "").strip()
    if fmt == "16:9":
        return "16:9"
    return "9:16"


def _reels_size(fmt: str):
    if (fmt or "").strip() == "16:9":
        return (1280, 720)
    return (720, 1280)


def _wrap_text(text: str, max_chars: int = 22):
    words = (text or "").split()
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
    total_frames = max(1, int(duration) * fps)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    lines = _wrap_text(text, max_chars=22)

    for i in range(total_frames):
        img = Image.new("RGB", (w, h), (10, 16, 28))
        draw = ImageDraw.Draw(img)

        box_margin = 60
        box_w = w - 2 * box_margin
        box_h = 70 * len(lines) + 80
        x0 = box_margin
        y0 = (h // 2) - (box_h // 2)
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


async def _openai_generate_image(prompt: str) -> Dict[str, Any]:
    """
    Genera imagen con OpenAI.
    Devuelve:
      - image_url (si el API devolvió url)
      - b64_png (si devolvió b64_json)
      - data (respuesta completa)
    """
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
                "prompt": prompt,
                "size": "1024x1024",
            },
        )

    data = r.json()

    # Intentar url primero
    image_url = None
    try:
        image_url = data.get("data", [{}])[0].get("url")
    except Exception:
        image_url = None

    # Intentar b64_json
    b64_png = None
    if not image_url:
        try:
            b64_png = data.get("data", [{}])[0].get("b64_json")
        except Exception:
            b64_png = None

    return {"status": "ok", "image_url": image_url, "b64_png": b64_png, "data": data}


def _save_b64_png_to_file(b64_png: str) -> Path:
    raw = base64.b64decode(b64_png)
    out_path = OUT_DIR / f"img_{uuid.uuid4().hex}.png"
    out_path.write_bytes(raw)
    return out_path


def _fal_upload_if_possible(file_path: Path) -> Optional[str]:
    """
    Sube archivo a FAL si el SDK lo soporta (muchas versiones tienen upload_file()).
    Devuelve una URL pública usable como image_url.
    """
    if hasattr(fal_client, "upload_file"):
        try:
            return fal_client.upload_file(str(file_path))
        except Exception:
            return None
    return None


# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}


# =========================
# UI (BONITA)
# =========================
@app.get("/app", response_class=HTMLResponse)
def app_ui():
    # Si existe ui.html, úsalo. Si no, usa la UI embebida.
    if os.path.exists("ui.html"):
        return open("ui.html", "r", encoding="utf-8").read()

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
    .badge{display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(124,58,237,.20);border:1px solid rgba(124,58,237,.35);font-size:12px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Corazón Studio AI ✨</div>
  <p class="sub">Chat, imágenes, reels MP4 y <b>video cine realista (texto → video)</b>.</p>

  <div class="grid">
    <div class="card">
      <h3>💬 Chat</h3>
      <textarea id="chatText" rows="3" placeholder="Escribe tu mensaje..."></textarea>
      <div style="height:10px"></div>
      <button id="chatBtn">Enviar</button>
      <div class="small">Tip: si no responde, revisa que el backend esté activo.</div>
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
      <span class="badge">Kling (FAL) · image-to-video</span>
      <div style="height:10px"></div>
      <textarea id="vidText" rows="3" placeholder="Ej: Una niña sonriente caminando en un parque al atardecer, estilo cinematográfico, luz cálida, cámara suave, realista"></textarea>
      <div style="height:10px"></div>
      <div class="row">
        <select id="vidDur">
          <option value="5">5s</option>
          <option value="10">10s</option>
        </select>
        <select id="vidFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>
      <div style="height:10px"></div>
      <div class="row">
        <input id="vidCfg" type="number" step="0.1" value="0.5" />
        <select id="vidQ">
          <option value="512p">Calidad 512p (más rápido)</option>
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
      const url = data.url || data?.data?.data?.[0]?.url; // soporta ambos formatos
      const b64 = data.b64_data_url;
      if(url){
        img.src = url;
        img.style.display="block";
      }else if(b64){
        img.src = b64;
        img.style.display="block";
      }else{
        info.textContent = "Respuesta recibida (sin url/b64 directo).";
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
      out.textContent = "Listo ✅ Descargando reels.mp4";
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
      if(data.video_url){
        vid.src = data.video_url;
        vid.style.display="block";
        out.textContent = "Video listo ✅";
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
                        "content": "Eres un asistente amable, profesional y empático. Hablas con calidez, respeto y claridad.",
                    },
                    {"role": "user", "content": req.message},
                ],
            },
        )

    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}


# =========================
# IMAGEN (devuelve url o b64_data_url)
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    res = await _openai_generate_image(req.prompt)

    if res.get("status") != "ok":
        return res

    if res.get("image_url"):
        return {"status": "ok", "url": res["image_url"], "data": res}

    if res.get("b64_png"):
        # para que el frontend lo muestre directo
        b64_data_url = "data:image/png;base64," + res["b64_png"]
        return {"status": "ok", "b64_data_url": b64_data_url, "data": res}

    return {"status": "error", "message": "OpenAI no devolvió image_url ni b64_json", "raw": res}


# =========================
# VIDEO (modo reels) - igual que antes
# =========================
class VideoRequest(BaseModel):
    mode: str  # reels
    text: str = ""
    duration: int = 6
    format: str = "9:16"


@app.post("/video")
async def video(req: VideoRequest):
    mode = (req.mode or "").strip().lower()

    if mode == "reels":
        out_name = f"reels_{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name
        _make_reels_mp4(req.text or " ", max(1, int(req.duration)), req.format, out_path)
        return FileResponse(path=str(out_path), media_type="video/mp4", filename=out_name)

    return {"status": "error", "message": "Modo de video no válido. Usa: reels"}


# =========================
# 🎬 VIDEO CINE REALISTA (TEXTO → VIDEO)
# 1) genera imagen guía con OpenAI
# 2) manda a FAL Kling image-to-video
# =========================
@app.post("/video-cine")
async def video_cine(req: VideoCineRequest):
    if not os.getenv("FAL_KEY"):
        return {"status": "error", "message": "FAL_KEY no está configurada en Render"}

    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OPENAI_API_KEY no está configurada en Render"}

    aspect_ratio = _aspect_ratio_from_format(req.format)

    # (A) Crear imagen guía (esto vuelve el resultado MUCHO mejor que solo texto)
    img_prompt = (
        f"{req.text}. "
        f"Frame cinematográfico realista, iluminación de cine, alto detalle, natural, sin texto, sin logos."
    )
    img_res = await _openai_generate_image(img_prompt)
    if img_res.get("status") != "ok":
        return img_res

    image_url = img_res.get("image_url")

    # Si no hay URL, viene b64_json => guardamos temporal y subimos a FAL
    if not image_url and img_res.get("b64_png"):
        try:
            img_path = _save_b64_png_to_file(img_res["b64_png"])
        except Exception as e:
            return {"status": "error", "message": "No se pudo guardar la imagen temporal", "detail": str(e)}

        uploaded = _fal_upload_if_possible(img_path)
        if not uploaded:
            return {
                "status": "error",
                "message": "No se pudo subir la imagen a FAL (tu SDK fal_client no trae upload_file).",
                "tip": "Solución: actualizar fal-client o usar un image_url directo.",
            }
        image_url = uploaded

    if not image_url:
        return {"status": "error", "message": "No se pudo obtener image_url desde OpenAI"}

    # (B) FAL: Kling image-to-video
    # Nota: duration en este modelo suele ser "5" o "10"
    duration = int(req.duration)
    if duration not in (5, 10):
        duration = 5

    try:
        result = fal_client.run(
            FAL_KLING_IMAGE_TO_VIDEO,
            arguments={
                "prompt": req.text,
                "image_url": image_url,
                "duration": str(duration),
                "aspect_ratio": aspect_ratio,
                "cfg_scale": float(req.cfg),
            },
        )
        video_url = _extract_video_url(result)
        if not video_url:
            return {"status": "error", "message": "No llegó video_url", "raw": result}

        return {
            "status": "ok",
            "video_url": video_url,
            "image_url_used": image_url,
        }
    except Exception as e:
        return {"status": "error", "message": "FAL error", "detail": str(e)}
