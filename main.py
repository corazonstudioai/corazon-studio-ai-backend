import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en Environment Variables (Render).")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_VIDEO_MODEL = os.environ.get("OPENAI_VIDEO_MODEL", "sora-2-pro")

ASSISTANT_PROMPT = os.environ.get(
    "ASSISTANT_PROMPT",
    "Eres un asistente amable, claro y práctico. Responde en el idioma del usuario."
)

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Corazon Studio AI API", version="1.0.0")

# CORS: permite que tu frontend llame al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego si quieres lo cerramos a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "es"


class ImageRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"


class VideoRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1280x720"
    seconds: Optional[int] = 8


# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "message": "API funcionando. Ve a /docs para probar endpoints."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Respuestas simples con Responses API
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "user", "content": req.message},
            ],
        )

        # El SDK puede devolver el texto en diferentes estructuras; esto es lo más robusto:
        text = getattr(response, "output_text", None)
        if not text:
            # fallback si output_text no existe
            text = str(response)

        return {"ok": True, "reply": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en /chat: {e}")


@app.post("/image")
def image(req: ImageRequest):
    try:
        # Genera imagen (retorna base64)
        img = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=req.prompt,
            size=req.size,
        )

        b64 = img.data[0].b64_json
        data_url = f"data:image/png;base64,{b64}"
        return {"ok": True, "image": data_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en /image: {e}")


@app.post("/video")
def video_start(req: VideoRequest):
    """
    Inicia render de video (asíncrono). Retorna video_id + status.
    """
    try:
        v = client.videos.create(
            model=OPENAI_VIDEO_MODEL,
            prompt=req.prompt,
            size=req.size,
            seconds=str(req.seconds),
        )
        return {"ok": True, "video_id": v.id, "status": v.status, "progress": getattr(v, "progress", None)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en /video: {e}")


@app.get("/video/{video_id}")
def video_status(video_id: str):
    """
    Consulta estado del video.
    """
    try:
        v = client.videos.retrieve(video_id)
        return {"ok": True, "video_id": v.id, "status": v.status, "progress": getattr(v, "progress", None)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en /video/{{id}}: {e}")


@app.get("/video/{video_id}/content")
def video_content(video_id: str):
    """
    Descarga/stream del MP4 cuando status == completed.
    Usa el endpoint oficial GET /videos/{video_id}/content.  (proxy)
    """
    try:
        url = f"https://api.openai.com/v1/videos/{video_id}/content"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

        with httpx.stream("GET", url, headers=headers, timeout=120) as r:
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.text)

            return StreamingResponse(r.iter_bytes(), media_type="video/mp4")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando video: {e}")
