import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# =========================
# CONFIG
# =========================
APP_TITLE = "Corazón Studio AI Backend"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_PROMPT = os.getenv(
    "ASSISTANT_PROMPT",
    "Eres un asistente útil, amable y claro. Responde en español y en pasos cuando te lo pidan.",
)

# Mientras pruebas, deja "*". Luego lo hacemos más seguro.
ALLOWED_ORIGINS = ["*"]


# =========================
# APP
# =========================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lee OPENAI_API_KEY desde Render (Environment Variables)
client = OpenAI()


# =========================
# MODELOS
# =========================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# =========================
# RUTAS
# =========================
@app.get("/")
def root():
    return {"mensaje": "Backend Corazón Studio AI activo"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Falta OPENAI_API_KEY en Render (Environment Variables).",
        )

    user_text = (body.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="El mensaje está vacío.")

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )

        reply_text = (response.output_text or "").strip()
        if not reply_text:
            reply_text = "No pude generar respuesta. Intenta de nuevo."

        return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error con OpenAI: {str(e)}")
