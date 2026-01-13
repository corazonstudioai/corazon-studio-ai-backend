import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI (oficial)
from openai import OpenAI

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Corazón Studio AI Backend")

# CORS (para GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción puedes limitar a tu dominio de GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Config desde Render Environment
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # puedes cambiarlo en Render
ASSISTANT_PROFILE = os.getenv("ASSISTANT_PROFILE", "")   # opcional: tu “estilo/reglas”

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------------
# Modelos
# -----------------------------
class ChatIn(BaseModel):
    message: str
    language: str = "es"   # "es" o "en"

class ChatOut(BaseModel):
    reply: str

# -----------------------------
# Rutas
# -----------------------------
@app.get("/")
def root():
    return {"mensaje": "Backend Corazón Studio AI activo ❤️", "ok": True}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    msg = (payload.message or "").strip()
    lang = (payload.language or "es").strip().lower()

    if not msg:
        return {"reply": "Escribe un mensaje primero." if lang == "es" else "Please type a message first."}

    if client is None:
        return {"reply": "Falta configurar OPENAI_API_KEY en Render." if lang == "es" else "OPENAI_API_KEY is missing on Render."}

    # Instrucciones: idioma + estilo (ASSISTANT_PROFILE opcional)
    if lang == "en":
        system_text = "You are Corazón Studio AI. Reply ONLY in English."
    else:
        system_text = "Eres Corazón Studio AI. Responde SOLO en Español."

    if ASSISTANT_PROFILE:
        system_text += "\n\n" + ASSISTANT_PROFILE

    # Usamos Responses API (recomendado actualmente) 0
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": msg},
        ],
    )

    # Extraer texto final (seguro)
    reply_text = ""
    try:
        reply_text = resp.output_text
    except Exception:
        reply_text = ""

    if not reply_text:
        reply_text = "No pude generar respuesta. Intenta de nuevo." if lang == "es" else "I couldn't generate a reply. Please try again."

    return {"reply": reply_text}
