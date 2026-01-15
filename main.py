import os
import time
import sqlite3
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en las variables de entorno de Render.")

OPENAI_BASE_URL = "https://api.openai.com/v1"

# DB (persistencia simple)
DB_PATH = os.getenv("DB_PATH", "jobs.db")

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_init():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS video_jobs (
            video_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            error TEXT,
            content BLOB
        )
    """)
    con.commit()
    con.close()

def db_upsert(video_id: str, status: str, progress: int, prompt: str, error: Optional[str] = None, content: Optional[bytes] = None):
    now = int(time.time())
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO video_jobs(video_id, status, progress, prompt, created_at, updated_at, error, content)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            status=excluded.status,
            progress=excluded.progress,
            prompt=excluded.prompt,
            updated_at=excluded.updated_at,
            error=excluded.error,
            content=COALESCE(excluded.content, video_jobs.content)
    """, (video_id, status, progress, prompt, now, now, error, content))
    con.commit()
    con.close()

def db_get(video_id: str) -> Optional[Dict[str, Any]]:
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT video_id, status, progress, prompt, created_at, updated_at, error, content FROM video_jobs WHERE video_id=?", (video_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "video_id": row[0],
        "status": row[1],
        "progress": row[2],
        "prompt": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "error": row[6],
        "content": row[7],
    }

# =========================
# App
# =========================
app = FastAPI(title="Corazon Studio AI API", version="1.0.0")
db_init()

# CORS (para tu web onrender)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # si quieres, luego lo cerramos a tu dominio web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = "es"

class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"

class VideoRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"ok": True, "service": "corazon-studio-ai-backend"}

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# CHAT (opcional)
# =========================
@app.post("/chat")
async def chat(req: ChatRequest):
    # Placeholder simple para no romper tu web si luego lo agregas
    return JSONResponse({"ok": True, "reply": f"Recibido: {req.message}"})

# =========================
# IMAGE (opcional)
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    # Placeholder simple (para no romper tu web si luego lo agregas)
    return JSONResponse({"ok": True, "note": "Endpoint /image listo. Conecta aquí tu generación real cuando quieras."})

# =========================
# VIDEO (real, con persistencia de estado)
# =========================
async def _openai_create_video(prompt: str) -> bytes:
    """
    IMPORTANTE:
    Aquí hacemos una llamada genérica.
    Si tu modelo/endpoint exacto cambia, lo ajustamos.
    Por ahora, esperamos que OpenAI devuelva un mp4 (bytes) o un URL.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": os.getenv("OPENAI_VIDEO_MODEL", "gpt-video-1"),
        "prompt": prompt,
        # algunos proveedores aceptan duration/size; si no, se ignora
        "duration": 5,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OPENAI_BASE_URL}/videos", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

        data = r.json()

        # Caso A: te regresa un URL directo del video
        if isinstance(data, dict) and "url" in data and data["url"]:
            vr = await client.get(data["url"])
            vr.raise_for_status()
            return vr.content

        # Caso B: te regresa base64
        if isinstance(data, dict) and "video_base64" in data and data["video_base64"]:
            import base64
            return base64.b64decode(data["video_base64"])

        # Caso C: estructura diferente
        raise RuntimeError("Respuesta de video no reconocida (no vino url ni base64).")

@app.post("/video")
async def video_start(req: VideoRequest):
    # Creamos un id local (persistente) para tu frontend
    video_id = f"video_{int(time.time())}_{os.urandom(6).hex()}"
    db_upsert(video_id, "queued", 0, req.prompt)

    # Procesamos en background simple (sin celery)
    # Nota: FastAPI no tiene background robusto en serverless; Render usualmente aguanta bien.
    async def run():
        try:
            db_upsert(video_id, "processing", 20, req.prompt)
            content = await _openai_create_video(req.prompt)
            db_upsert(video_id, "completed", 100, req.prompt, content=content)
        except Exception as e:
            db_upsert(video_id, "failed", 0, req.prompt, error=str(e))

    # Lanzar tarea async
    import asyncio
    asyncio.create_task(run())

    return {"ok": True, "video_id": video_id, "status": "queued", "progress": 0}

@app.get("/video/{video_id}")
def video_status(video_id: str):
    job = db_get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="video_id no existe.")
    return {
        "ok": True,
        "video_id": job["video_id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
    }

@app.get("/video/{video_id}/content")
def video_content(video_id: str):
    job = db_get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="video_id no existe.")
    if job["status"] != "completed" or not job["content"]:
        raise HTTPException(status_code=409, detail="El video todavía no está listo.")
    # mp4 bytes
    return Response(content=job["content"], media_type="video/mp4")
