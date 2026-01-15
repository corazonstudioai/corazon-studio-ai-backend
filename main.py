import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# FAL se usa para Runway y Pika
FAL_KEY = os.getenv("FAL_KEY")

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELOS
# =========================
class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    provider: str  # "runway" | "pika"
    prompt: str
    duration: int = 6

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Corazón Studio AI backend activo"
    }

# =========================
# CHAT
# =========================
@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente profesional, amable, empático y respetuoso. "
                            "Hablas con claridad, paciencia y calidez."
                        )
                    },
                    {"role": "user", "content": req.message}
                ]
            }
        )
    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"]}

# =========================
# IMÁGENES
# =========================
@app.post("/image")
async def image(req: ImageRequest):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": req.prompt,
                "size": "1024x1024"
            }
        )
    return r.json()

# =========================
# VIDEO – RUNWAY / PIKA
# =========================
@app.post("/video")
def video(req: VideoRequest):

    if not FAL_KEY:
        return {"error": "FAL_KEY no configurada en Render"}

    provider = req.provider.lower()

    # ---- RUNWAY
    if provider == "runway":
        result = fal_client.run(
            "runwayml/gen2",
            arguments={
                "prompt": req.prompt,
                "seconds": req.duration
            }
        )
        return {
            "status": "ok",
            "provider": "runway",
            "result": result
        }

    # ---- PIKA
    if provider == "pika":
        result = fal_client.run(
            "pika/video",
            arguments={
                "prompt": req.prompt,
                "seconds": req.duration
            }
        )
        return {
            "status": "ok",
            "provider": "pika",
            "result": result
        }

    return {"error": "Proveedor no válido. Usa runway o pika"}
