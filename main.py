import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import fal_client

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# Fal (para Runway/Pika vía fal_client)
FAL_KEY = os.getenv("FAL_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")  # (si los usas directo, aquí quedan)
PIKA_API_KEY = os.getenv("PIKA_API_KEY")

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
    mode: str  # reels | runway | pika
    text: str = ""
    duration: int = 6
    format: str = "9:16"

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Corazón Studio AI backend activo"}

# =========================
# CHAT (AMABLE + EMPÁTICO)
# =========================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        return {"error": "Falta OPENAI_API_KEY en variables de entorno."}

    payload = {
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
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return {"reply": data["choices"][0]["message"]["content"]}
    except httpx.HTTPStatusError as e:
        return {"error": f"OpenAI error: {e.response.status_code}", "details": e.response.text}
    except Exception as e:
        return {"error": "Error en /chat", "details": str(e)}

# =========================
# IMÁGENES
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    if not OPENAI_API_KEY:
        return {"error": "Falta OPENAI_API_KEY en variables de entorno."}

    try:
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
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"OpenAI error: {e.response.status_code}", "details": e.response.text}
    except Exception as e:
        return {"error": "Error en /image", "details": str(e)}

# =========================
# VIDEO
# =========================
@app.post("/video")
async def video(req: VideoRequest):
    # -------- MODO A: REELS (texto animado simple)
    if req.mode == "reels":
        return {
            "status": "ok",
            "type": "reels",
            "message": "Video tipo reels generado (demo).",
            "text": req.text,
            "duration": req.duration,
            "format": req.format,
        }

    # Para Runway/Pika vía fal_client necesitas FAL_KEY en variables
    if not FAL_KEY:
        return {"error": "Falta FAL_KEY para usar Runway/Pika vía fal_client."}

    # -------- MODO B: RUNWAY (vía fal)
    if req.mode == "runway":
        try:
            result = fal_client.run(
                "runwayml/gen2",
                arguments={"prompt": req.text, "seconds": req.duration},
            )
            return {"status": "ok", "provider": "runway", "result": result}
        except Exception as e:
            return {"error": "Error Runway (fal)", "details": str(e)}

    # -------- MODO C: PIKA (vía fal)
    if req.mode == "pika":
        try:
            result = fal_client.run(
                "pika/video",
                arguments={"prompt": req.text, "seconds": req.duration},
            )
            return {"status": "ok", "provider": "pika", "result": result}
        except Exception as e:
            return {"error": "Error Pika (fal)", "details": str(e)}

    return {"error": "Modo de video no válido. Usa: reels | runway | pika"}
