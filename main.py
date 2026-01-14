# principal.py
import os
import time
import base64
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# Config (Render ENV Vars)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

# Nota:
# - Para video, este backend usa un "job" y devuelve links para consultar.
# - Si todavía no tienes un modelo/endpoint real de video, igual te funcionará el flujo
#   (start/status/content), y tu frontend podrá mostrar link/estado sin crashear.
OPENAI_VIDEO_MODEL = os.getenv("OPENAI_VIDEO_MODEL", "").strip()  # opcional

# ============================================================
# App
# ============================================================
app = FastAPI(title="Corazon Studio AI API", version="1.0.0")

# CORS (para que tu web en Render pueda llamar al backend sin "Failed to fetch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # si quieres más seguro: pon tu dominio exacto
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Schemas
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    language: str = Field("es")

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    size: str = Field("1024x1024")

class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

# ============================================================
# Helpers
# ============================================================
def _require_key():
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Falta OPENAI_API_KEY en variables de entorno (Render).",
        )

async def _openai_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        # Si OpenAI devuelve error, lo pasamos tal cual
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=r.text)
        return r.json()

def _as_data_url(mime: str, b64: str) -> str:
    return f"data:{mime};base64,{b64}"

# ============================================================
# Simple in-memory video jobs (demo)
# ============================================================
VIDEO_JOBS: Dict[str, Dict[str, Any]] = {}

def _new_video_id() -> str:
    return f"video_{int(time.time()*1000)}"

# ============================================================
# Routes
# ============================================================
@app.get("/")
def root():
    return {"ok": True, "message": "API funcionando. Ve a /docs para probar endpoints."}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(body: ChatRequest):
    _require_key()

    # Usamos Responses API (texto)
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "input": body.message,
    }

    data = await _openai_post_json("https://api.openai.com/v1/responses", payload)

    # response.output_text suele venir ya listo
    text = data.get("output_text")
    if not text:
        # fallback simple si cambia formato
        text = str(data)

    return {"ok": True, "text": text}

@app.post("/image")
async def image(body: ImageRequest):
    _require_key()

    # Images API - devolvemos data URL para que el frontend la ponga en <img src="...">
    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": body.prompt,
        "size": body.size,
    }

    data = await _openai_post_json("https://api.openai.com/v1/images/generations", payload)

    # Esperado: data["data"][0]["b64_json"]
    try:
        b64 = data["data"][0]["b64_json"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Respuesta inesperada de imagen: {data}")

    return {
        "ok": True,
        "image": _as_data_url("image/png", b64),
    }

@app.post("/video")
async def video_start(body: VideoRequest):
    """
    Inicia un "job" de video.
    Devuelve video_id y status.
    """
    _require_key()

    video_id = _new_video_id()
    VIDEO_JOBS[video_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": body.prompt,
        "result": None,  # aquí guardaremos un "url" o "data"
        "error": None,
    }

    # Si en el futuro conectas un endpoint real de video, lo harías aquí:
    # - Guardas en VIDEO_JOBS[video_id] el "provider_id"
    # - Dejas status queued y luego /video/{id} consulta el estado real

    return {"ok": True, "video_id": video_id, "status": "queued", "progress": 0}

@app.get("/video/{video_id}")
async def video_status(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="video_id no existe.")

    # Demo: simulamos que termina rápido
    if job["status"] in ("queued", "processing"):
        job["status"] = "completed"
        job["progress"] = 100

        # Resultado demo: NO generamos mp4 real aquí (sin librerías de video).
        # Te devolvemos un link estable a /video/{id}/content
        job["result"] = {"content_url": f"/video/{video_id}/content"}

    return {
        "ok": True,
        "video_id": video_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    }

@app.get("/video/{video_id}/content")
async def video_content(video_id: str):
    """
    Devuelve el contenido o link final.
    En demo, devolvemos un mensaje + (opcional) una mini previsualización como imagen.
    """
    job = VIDEO_JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="video_id no existe.")

    if job["status"] != "completed":
        return {"ok": True, "status": job["status"], "message": "Aún no está listo."}

    # Demo: devolvemos un "preview" imagen usando el prompt (opcional)
    # Esto evita que el frontend quede en negro si no hay video real todavía.
    preview = None
    try:
        payload = {
            "model": OPENAI_IMAGE_MODEL,
            "prompt": f"Fotograma del video: {job['prompt']}. Estilo realista.",
            "size": "1024x1024",
        }
        data = await _openai_post_json("https://api.openai.com/v1/images/generations", payload)
        b64 = data["data"][0]["b64_json"]
        preview = _as_data_url("image/png", b64)
    except Exception:
        preview = None

    return {
        "ok": True,
        "status": "completed",
        "message": "Video completado (demo). Conecta un modelo real de video cuando lo tengas.",
        "preview_image": preview,
        "download_url": None,  # aquí iría el link del mp4 real cuando lo conectes
    }
