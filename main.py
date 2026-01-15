import os
import re
import uuid
import time
import base64
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# =========================
# Config
# =========================
APP_NAME = "Corazon Studio AI Backend"
TMP_DIR = "/tmp/files"
os.makedirs(TMP_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

ASSISTANT_PROMPT = os.getenv(
    "ASSISTANT_PROMPT",
    (
        "Eres un asistente en español, amable, profesional y empático. "
        "Respondes claro y directo. Si te piden pasos, los das 1x1. "
        "No inventes información; si falta algo, dilo y ofrece alternativas."
    ),
).strip()

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "").strip()
FAL_KEY = os.getenv("FAL_KEY", "").strip()

# Runway (ajustable si tu cuenta usa otro dominio/version)
RUNWAY_BASE = os.getenv("RUNWAY_BASE", "https://api.dev.runwayml.com").strip()
RUNWAY_VERSION = os.getenv("RUNWAY_VERSION", "2024-11-06").strip()
RUNWAY_MODEL = os.getenv("RUNWAY_TEXT2VIDEO_MODEL", "gen3a_turbo").strip()

# Fal/Pika (modelo por defecto; puedes cambiarlo en variables si quieres)
PIKA_FAL_MODEL = os.getenv("PIKA_FAL_MODEL", "fal-ai/pika/v2/turbo/text-to-video").strip()

# Jobs en memoria (Render free reinicia a veces; es normal)
JOBS: Dict[str, Dict[str, Any]] = {}

# =========================
# App
# =========================
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo cerramos a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos (imágenes b64 guardadas)
app.mount("/files", StaticFiles(directory=TMP_DIR), name="files")


@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "openai_key": bool(OPENAI_API_KEY),
        "runway_key": bool(RUNWAY_API_KEY),
        "fal_key": bool(FAL_KEY),
    }


# =========================
# Helpers
# =========================
def _find_video_url(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            u = _find_video_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for it in obj:
            u = _find_video_url(it)
            if u:
                return u
    elif isinstance(obj, str):
        if re.search(r"https?://.*\.(mp4|webm)(\?.*)?$", obj, re.IGNORECASE):
            return obj
    return None


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


# =========================
# CHAT (OpenAI)
# =========================
@app.post("/chat")
def chat(payload: Dict[str, Any]):
    msg = (payload.get("message") or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message vacío")

    if not OPENAI_API_KEY:
        return {"reply": f"Recibí: {msg}\n\n(Nota: Falta OPENAI_API_KEY en Render.)"}

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": ASSISTANT_PROMPT},
            {"role": "user", "content": msg},
        ],
        "temperature": 0.6,
    }

    r = requests.post(url, headers=headers, json=data, timeout=90)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenAI chat error: {r.text}")

    j = r.json()
    reply = j["choices"][0]["message"]["content"]
    return {"reply": reply}


# =========================
# IMÁGENES (OpenAI o fallback)
# =========================
@app.post("/image")
def image(payload: Dict[str, Any], request: Request):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt vacío")

    # Fallback si no hay OpenAI key
    if not OPENAI_API_KEY:
        # imagen demo siempre visible
        return {"image_url": "https://picsum.photos/1024/1024"}

    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "size": "1024x1024",
    }

    r = requests.post(url, headers=headers, json=data, timeout=120)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenAI image error: {r.text}")

    j = r.json()
    d0 = (j.get("data") or [{}])[0]

    # Si devuelve URL directa:
    if isinstance(d0, dict) and d0.get("url"):
        return {"image_url": d0["url"]}

    # Si devuelve base64:
    if isinstance(d0, dict) and d0.get("b64_json"):
        img_id = str(uuid.uuid4())
        out_path = os.path.join(TMP_DIR, f"{img_id}.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(d0["b64_json"]))
        return {"image_url": f"{_base_url(request)}/files/{img_id}.png"}

    raise HTTPException(status_code=502, detail="No pude leer respuesta de imagen")


# =========================
# VIDEO REALISTA (Runway / Pika-Fal) -> Jobs
# =========================
@app.post("/video/realistic/start")
def video_realistic_start(payload: Dict[str, Any]):
    provider = (payload.get("provider") or "").strip().lower()
    prompt = (payload.get("prompt") or "").strip()
    ratio = (payload.get("ratio") or "1080:1920").strip()
    duration = int(payload.get("duration") or 5)

    if provider not in ("runway", "pika"):
        raise HTTPException(status_code=400, detail="provider inválido: runway o pika")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt vacío")

    job_id = str(uuid.uuid4())

    if provider == "runway":
        if not RUNWAY_API_KEY:
            raise HTTPException(status_code=400, detail="Falta RUNWAY_API_KEY en Render")

        create_url = f"{RUNWAY_BASE}/v1/text_to_video"
        headers = {
            "Authorization": f"Bearer {RUNWAY_API_KEY}",
            "X-Runway-Version": RUNWAY_VERSION,
            "Content-Type": "application/json",
        }
        body = {
            "model": RUNWAY_MODEL,
            "promptText": prompt,
            "ratio": ratio,
            "duration": duration,
        }

        r = requests.post(create_url, headers=headers, json=body, timeout=90)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Runway start error: {r.text}")

        task_id = r.json().get("id")
        if not task_id:
            raise HTTPException(status_code=502, detail=f"Runway no devolvió id: {r.text}")

        JOBS[job_id] = {"provider": "runway", "task_id": task_id}
        return {"job_id": job_id, "status": "STARTED"}

    # provider == pika (via Fal queue)
    if not FAL_KEY:
        raise HTTPException(status_code=400, detail="Falta FAL_KEY en Render")

    # Fal queue submit
    submit_url = f"https://queue.fal.run/{PIKA_FAL_MODEL}"
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    body = {
        "prompt": prompt,
        "duration": duration,
        # algunos modelos aceptan aspect_ratio / ratio; lo mandamos en forma común:
        "aspect_ratio": "9:16" if ratio.endswith(":1920") or ratio.endswith(":1280") else "16:9",
    }

    r = requests.post(submit_url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Fal submit error: {r.text}")

    j = r.json()
    status_url = j.get("status_url")
    response_url = j.get("response_url")
    if not status_url or not response_url:
        raise HTTPException(status_code=502, detail=f"Fal no devolvió status/response: {j}")

    JOBS[job_id] = {"provider": "pika", "status_url": status_url, "response_url": response_url}
    return {"job_id": job_id, "status": "STARTED"}


@app.get("/video/realistic/status/{job_id}")
def video_realistic_status(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job_id no encontrado")

    provider = info.get("provider")

    if provider == "runway":
        task_id = info["task_id"]
        url = f"{RUNWAY_BASE}/v1/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {RUNWAY_API_KEY}",
            "X-Runway-Version": RUNWAY_VERSION,
        }

        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code >= 400:
            return {"status": "RUNNING"}  # no matamos por un tick

        data = r.json()
        st = (data.get("status") or "").upper()

        video_url = None
        out = data.get("output")
        if isinstance(out, list) and out and isinstance(out[0], str):
            video_url = out[0]
        if not video_url:
            video_url = _find_video_url(data)

        if st in ("SUCCEEDED", "COMPLETED") and video_url:
            return {"status": "DONE", "video_url": video_url}
        if st in ("FAILED", "CANCELED"):
            return {"status": "ERROR", "detail": data}

        return {"status": "RUNNING"}

    if provider == "pika":
        headers = {"Authorization": f"Key {FAL_KEY}"}

        # poll status_url
        sr = requests.get(info["status_url"], headers=headers, timeout=60)
        if sr.status_code >= 400:
            return {"status": "RUNNING"}

        sj = sr.json()
        st = (sj.get("status") or "").lower()

        if st in ("completed", "succeeded", "success"):
            rr = requests.get(info["response_url"], headers=headers, timeout=60)
            if rr.status_code >= 400:
                return {"status": "ERROR", "detail": rr.text}

            out = rr.json()
            video_url = _find_video_url(out)
            if not video_url:
                # algunos devuelven dict con video.url
                v = out.get("video") if isinstance(out, dict) else None
                if isinstance(v, dict):
                    video_url = v.get("url")

            if not video_url:
                return {"status": "ERROR", "detail": out}

            return {"status": "DONE", "video_url": video_url}

        if st in ("failed", "error", "canceled"):
            return {"status": "ERROR", "detail": sj}

        return {"status": "RUNNING"}

    return {"status": "ERROR", "detail": "Provider desconocido"}
