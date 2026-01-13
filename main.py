import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ✅ CORS (para GitHub Pages / Render Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego si quieres lo cerramos a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelos (puedes cambiarlos en Render -> Environment)
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
VIDEO_MODEL = os.getenv("OPENAI_VIDEO_MODEL", "sora-2-pro")

# Prompt “Asistente Pro”
ASSISTANT_PROMPT = (
    os.getenv("ASSISTANT_PROMPT")
    or os.getenv("ASSISTANT_PROFILE")
    or os.getenv("ASSISTANT_PRO")
    or "Eres Corazón Studio AI: amable, claro y útil."
)

class ChatRequest(BaseModel):
    message: str
    lang: str = "es"  # "es" o "en"

class ImageRequest(BaseModel):
    prompt: str
    lang: str = "es"

class VideoRequest(BaseModel):
    prompt: str
    lang: str = "es"

def language_rule(lang: str) -> str:
    # ✅ Solo 1 idioma, no ambos
    if (lang or "").lower().startswith("en"):
        return "Reply ONLY in English. Do not use Spanish."
    return "Responde SOLO en español. No uses inglés."

@app.get("/health")
def health():
    return {"ok": True, "mensaje": "Backend Corazón Studio AI activo 💙"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    system = (
        f"{ASSISTANT_PROMPT}\n\n"
        f"{language_rule(req.lang)}\n"
        "Sé breve, amable y directo. Si falta información, pregunta 1 cosa a la vez."
    )

    try:
        r = client.responses.create(
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": req.message},
            ],
        )

        # output_text existe en el SDK moderno
        reply = getattr(r, "output_text", None)
        if not reply:
            # fallback por si cambia el formato
            reply = "Listo ✅ (pero no pude leer el texto de salida)."

        return {"ok": True, "reply": reply.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")

@app.post("/image")
def image(req: ImageRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt vacío")

    # Le pasamos instrucción de idioma dentro del prompt si quieres textos en la imagen
    prompt = f"{req.prompt}\n\n{language_rule(req.lang)}"

    try:
        img = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
        )

        b64 = img.data[0].b64_json
        data_url = f"data:image/png;base64,{b64}"
        return {"ok": True, "data_url": data_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en imagen: {str(e)}")

@app.post("/video")
def video(req: VideoRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt vacío")

    # Prompt con regla de idioma (si el video incluye texto/voz)
    prompt = f"{req.prompt}\n\n{language_rule(req.lang)}"

    try:
        # Nota: video puede tardar, pero así lo entregamos “en una sola”
        v = client.videos.create_and_poll(
            model=VIDEO_MODEL,
            prompt=prompt,
            size="1280x720",
            seconds=6,
        )

        # v.id sirve para descargar contenido después
        return {"ok": True, "video_id": v.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en video: {str(e)}")

@app.get("/video/{video_id}")
def video_content(video_id: str):
    try:
        # Descarga el mp4 desde OpenAI (sin exponer tu llave al navegador)
        content = client.videos.content(video_id)  # bytes
        return Response(content=content, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error bajando video: {str(e)}")
