import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ----------------------------
# Configuración
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

ASSISTANT_ID = (
    os.getenv("ASSISTANT_PRO_ID")
    or os.getenv("ASSISTANT_PRO")
    or os.getenv("ASSISTANT_ID")
    or ""
).strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Corazón Studio AI Backend")

# ----------------------------
# CORS (GitHub Pages)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo hacemos más seguro
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Modelos de datos
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    language: str = "es"  # "es" o "en"

class ChatResponse(BaseModel):
    reply: str

class VideoPlanRequest(BaseModel):
    idea: str                 # texto del usuario
    language: str = "es"      # "es" o "en"
    duration_seconds: int = 60
    format: str = "9:16"      # "9:16" reels, "16:9" horizontal
    vibe: str = "profesional y cálido"  # tono general (puedes cambiarlo)

class VideoPlanResponse(BaseModel):
    plan: str  # por ahora devolvemos texto súper organizado (luego lo hacemos JSON si quieres)


# ----------------------------
# Rutas básicas
# ----------------------------
@app.get("/")
def root():
    return {"mensaje": "Backend Corazón Studio AI activo ❤️", "ok": True}

@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------
# Prompts por idioma
# ----------------------------
def get_system_prompt(language: str) -> str:
    if language.lower() == "en":
        return (
            "You are Corazón Studio AI. Create clear, practical outputs. "
            "Respond ONLY in English."
        )
    return (
        "Eres Corazón Studio AI. Crea resultados claros y prácticos. "
        "Responde SOLO en español."
    )

# ----------------------------
# Lógica IA (con o sin Assistant Pro)
# ----------------------------
def call_ai(user_text: str, language: str) -> str:
    if not client:
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en Render.")

    system_prompt = get_system_prompt(language)

    # Si hay Assistant Pro, lo usamos
    if ASSISTANT_ID:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_text,
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            additional_instructions=system_prompt,
        )

        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status in ("completed", "failed", "cancelled", "expired"):
                break
            time.sleep(0.6)

        if run.status != "completed":
            raise HTTPException(status_code=500, detail=f"Assistant run status: {run.status}")

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for m in messages.data:
            if m.role == "assistant":
                parts = []
                for block in m.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text.value)
                text = "\n".join(parts).strip()
                if text:
                    return text

        raise HTTPException(status_code=500, detail="No se recibió respuesta del assistant.")

    # Fallback sin assistant (modelo directo)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Endpoint Chat
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message vacío")

    reply = call_ai(msg, req.language)
    return ChatResponse(reply=reply)


# ----------------------------
# NUEVO: Endpoint Video Plan (Opción 1)
# ----------------------------
@app.post("/video-plan", response_model=VideoPlanResponse)
def video_plan(req: VideoPlanRequest):
    idea = (req.idea or "").strip()
    if not idea:
        raise HTTPException(status_code=400, detail="idea vacía")

    # Seguridad básica
    duration = req.duration_seconds
    if duration < 10:
        duration = 10
    if duration > 60:
        duration = 60

    prompt = f"""
Quiero que generes un PLAN DE VIDEO listo para producir desde esta idea:

IDEA:
{idea}

REQUISITOS:
- Duración total: {duration} segundos
- Formato: {req.format}
- Estilo/Vibe: {req.vibe}
- Devuelve TODO súper ordenado con estos encabezados exactos:

1) TÍTULO DEL VIDEO (corto y llamativo)
2) OBJETIVO (1 línea)
3) GUION NARRADO COMPLETO (para voz)
4) ESCENAS (lista numerada). Para cada escena incluye:
   - Duración en segundos
   - Texto en pantalla (máximo 8 palabras)
   - Narración de esa escena (1–2 líneas)
   - PROMPT DE IMAGEN (descripción clara para generar imagen)
   - Sugerencia de ambiente/música (1 frase)
5) CTA FINAL (llamado a la acción)
6) NOTAS (si recomiendas algo para que se vea profesional)

Reglas:
- No uses dos idiomas: responde SOLO en el idioma elegido.
- Hazlo listo para Reels/ads.
"""

    text = call_ai(prompt, req.language)
    return VideoPlanResponse(plan=text)
