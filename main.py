import os
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno (Render).")

VIDEO_MODEL = os.getenv("OPENAI_VIDEO_MODEL", "sora-2")
VIDEO_SIZE = os.getenv("OPENAI_VIDEO_SIZE", "1280x720")  # recomendado por docs
DEFAULT_SECONDS = int(os.getenv("OPENAI_VIDEO_SECONDS", "5"))

# OJO: pon aquí tu web de Render para permitir CORS
ALLOWED_ORIGINS = [
    "https://corazon-studio-ai-web.onrender.com",
    "http://localhost:5173",
    "http://localhost:3000",
]

OPENAI_BASE = "https://api.openai.com/v1"

app = FastAPI(title="Corazon Studio AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Schemas
# =========================
class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    seconds: Optional[int] = Field(default=None, ge=1, le=20)
    size: Optional[str] = None  # ejemplo: "1280x720"


# =========================
# Helpers
# =========================
def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


# =========================
# Health / Root
# =========================
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "corazon-studio-ai-backend"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


# =========================
# VIDEO (Sora) — FIX REAL
# =========================
@app.post("/video")
async def video_start(payload: VideoRequest) -> Dict[str, Any]:
    """
    Crea un job async de video.
    IMPORTANTE: La API usa `seconds`, NO `duration`. 1
    """
    seconds = payload.seconds or DEFAULT_SECONDS
    size = payload.size or VIDEO_SIZE

    # La API de /videos requiere multipart/form-data (prompt/model/size/seconds). 2
    form = {
        "prompt": payload.prompt,
        "model": VIDEO_MODEL,
        "size": size,
        "seconds": str(seconds),
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{OPENAI_BASE}/videos",
            headers=_auth_headers(),
            files=form,  # multipart/form-data
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=r.text)

    data = r.json()
    return {
        "ok": True,
        "video_id": data.get("id"),
        "status": data.get("status"),
        "progress": data.get("progress", 0),
        "raw": data,
    }


@app.get("/video/{video_id}")
async def video_status(video_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(
            f"{OPENAI_BASE}/videos/{video_id}",
            headers={**_auth_headers(), "accept": "application/json"},
        )

    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="video_id no existe.")
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=r.text)

    data = r.json()
    return {
        "ok": True,
        "video_id": data.get("id"),
        "status": data.get("status"),
        "progress": data.get("progress", 0),
        "error": data.get("error"),
        "raw": data,
    }


@app.get("/video/{video_id}/content")
async def video_content(video_id: str):
    """
    Descarga el MP4 cuando el estado sea completed.
    Docs: GET /videos/{video_id}/content 3
    """
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(
            f"{OPENAI_BASE}/videos/{video_id}/content",
            headers=_auth_headers(),
            follow_redirects=True,
        )

    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="video_id no existe.")
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=r.text)

    # Stream del mp4
    content_type = r.headers.get("content-type", "video/mp4")
    return StreamingResponse(iter([r.content]), media_type=content_type)
