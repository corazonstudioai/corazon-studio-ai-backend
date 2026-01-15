import os
import uuid
import re
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional: fal.ai client (for Pika via Fal)
try:
    import fal_client
except Exception:
    fal_client = None


app = FastAPI(title="Corazón Studio AI API")

# CORS (para tu frontend en onrender)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción puedes limitarlo a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helpers
# ----------------------------

def _find_video_url(obj: Any) -> Optional[str]:
    """Busca cualquier URL de video dentro de un dict/list (mp4/webm)."""
    if isinstance(obj, dict):
        for v in obj.values():
            u = _find_video_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for item in obj:
            u = _find_video_url(item)
            if u:
                return u
    elif isinstance(obj, str):
        if re.search(r"https?://.*\.(mp4|webm)(\?.*)?$", obj, re.IGNORECASE):
            return obj
    return None


# ----------------------------
# Chat (demo simple)
# ----------------------------

class ChatIn(BaseModel):
    message: str

@app.post("/chat")
async def chat(payload: ChatIn):
    # Aquí luego conectas OpenAI si quieres.
    return {"reply": f"Recibí: {payload.message}"}


# ----------------------------
# Imagen (demo simple)
# ----------------------------

class ImageIn(BaseModel):
    prompt: str

@app.post("/image")
async def image(payload: ImageIn):
    """
    DEMO: para que tu UI no falle.
    Si luego conectas OpenAI Images, aquí reemplazas.
    """
    # Imagen placeholder (no IA) para que SIEMPRE se vea algo
    # Puedes cambiarlo luego por tu generador real.
    return {
        "url": "https://picsum.photos/768/768",
        "note": "Demo image (placeholder). Conecta tu generador real cuando quieras."
    }


# ----------------------------
# Video - Modo A (Reels Texto Animado)
# (Se genera EN el frontend con Canvas + MediaRecorder)
# Backend NO necesita hacer nada.
# ----------------------------


# ----------------------------
# Video - Modo B (Realista IA)
# Providers: runway | pika
# ----------------------------

JOBS: Dict[str, Dict[str, Any]] = {}  # job_id -> info


class RealisticVideoIn(BaseModel):
    provider: str  # "runway" o "pika"
    prompt: str
    ratio: str = "1080:1920"
    duration: int = 5


@app.post("/video/realistic/start")
async def realistic_video_start(payload: RealisticVideoIn):
    provider = payload.provider.lower().strip()
    prompt = payload.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="El prompt está vacío.")

    job_id = str(uuid.uuid4())

    if provider == "runway":
        runway_key = os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")
        if not runway_key:
            raise HTTPException(
                status_code=400,
                detail="Falta RUNWAY_API_KEY (o RUNWAYML_API_SECRET) en Render Environment Variables."
            )

        # Runway API: POST /v1/text_to_video con Authorization: Bearer + X-Runway-Version=2024-11-06  0
        runway_model = os.getenv("RUNWAY_TEXT2VIDEO_MODEL", "veo3.1")  # valores aceptados en docs 1

        url = "https://api.dev.runwayml.com/v1/text_to_video"
        headers = {
            "Authorization": f"Bearer {runway_key}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        }
        body = {
            "model": runway_model,
            "promptText": prompt,
            "ratio": payload.ratio,
            "duration": payload.duration,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Runway error: {r.text}")

        data = r.json()
        task_id = data.get("id")
        if not task_id:
            raise HTTPException(status_code=502, detail=f"Runway no devolvió task id: {data}")

        JOBS[job_id] = {"provider": "runway", "task_id": task_id}
        return {"job_id": job_id, "status": "STARTED"}

    elif provider == "pika":
        # Pika API se usa vía Fal.ai 2
        if fal_client is None:
            raise HTTPException(status_code=500, detail="fal-client no está instalado. Revisa requirements.txt.")

        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise HTTPException(
                status_code=400,
                detail="Falta FAL_KEY en Render Environment Variables (Fal.ai)."
            )

        # Usamos modelo turbo para velocidad 3
        model_id = os.getenv("PIKA_FAL_MODEL", "fal-ai/pika/v2/turbo/text-to-video")

        # Nota: Fal puede tardar. Aquí lo lanzamos como job y lo resolvemos con /status.
        # Guardamos el prompt para correrlo en el status si no existe request_id.
        JOBS[job_id] = {"provider": "pika", "model": model_id, "prompt": prompt}
        return {"job_id": job_id, "status": "STARTED"}

    else:
        raise HTTPException(status_code=400, detail="Provider inválido. Usa: runway o pika.")


@app.get("/video/realistic/status/{job_id}")
async def realistic_video_status(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job_id no encontrado.")

    provider = info["provider"]

    if provider == "runway":
        runway_key = os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")
        if not runway_key:
            raise HTTPException(status_code=400, detail="Falta RUNWAY_API_KEY en Render.")

        task_id = info["task_id"]
        url = f"https://api.dev.runwayml.com/v1/tasks/{task_id}"  # GET task detail 4
        headers = {
            "Authorization": f"Bearer {runway_key}",
            "X-Runway-Version": "2024-11-06",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, headers=headers)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Runway status error: {r.text}")

        data = r.json()
        status = (data.get("status") or "").upper()

        # Muchos tasks devuelven output con URLs cuando finaliza
        video_url = None
        out = data.get("output")
        if isinstance(out, list) and out:
            # suele venir lista de URLs
            if isinstance(out[0], str):
                video_url = out[0]
        if not video_url:
            video_url = _find_video_url(data)

        if status in ("SUCCEEDED", "COMPLETED") and video_url:
            return {"status": "DONE", "video_url": video_url}

        if status in ("FAILED", "CANCELED"):
            return {"status": "ERROR", "detail": data}

        return {"status": "RUNNING"}

    if provider == "pika":
        if fal_client is None:
            raise HTTPException(status_code=500, detail="fal-client no está instalado.")
        if not os.getenv("FAL_KEY"):
            raise HTTPException(status_code=400, detail="Falta FAL_KEY en Render.")

        # Para evitar depender de endpoints internos de queue,
        # hacemos una sola ejecución cuando preguntas status por primera vez,
        # y guardamos el resultado.
        if "video_url" in info:
            return {"status": "DONE", "video_url": info["video_url"]}

        model_id = info["model"]
        prompt = info["prompt"]

        # Ejecuta Fal (bloquea hasta terminar). Para clips cortos suele funcionar bien.
        # fal_client.run() está documentado en PyPI. 5
        try:
            result = fal_client.run(model_id, arguments={"prompt": prompt})
        except Exception as e:
            return {"status": "ERROR", "detail": str(e)}

        video_url = _find_video_url(result)
        if not video_url:
            return {"status": "ERROR", "detail": result}

        info["video_url"] = video_url
        JOBS[job_id] = info
        return {"status": "DONE", "video_url": video_url}

    return {"status": "ERROR", "detail": "Provider desconocido."}


@app.get("/")
async def root():
    return {"ok": True, "service": "Corazón Studio AI API"}
