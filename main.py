
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI (SDK v1)
from openai import OpenAI

app = FastAPI(title="Corazón Studio AI Backend")

# =========================
# CORS (para GitHub Pages)
# =========================
# Puedes dejar "*" por ahora (rápido para pruebas).
# Luego lo hacemos más seguro con tu URL exacta.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Si tienes una variable tipo "ASSISTANT_PRO..." en Render,
# aquí soportamos varios nombres por si la guardaste distinto.
ASSISTANT_ID = (
    os.getenv("OPENAI_ASSISTANT_ID", "").strip()
    or os.getenv("ASSISTANT_PRO_ID", "").strip()
    or os.getenv("ASSISTANT_PRO", "").strip()
)

# Modelo por defecto si NO usas Assistants
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Schemas
# =========================
class ChatIn(BaseModel):
    message: str
    language: str = "es"  # "es" o "en"


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"mensaje": "Backend Corazón Studio AI activo ❤️", "ok": True}


@app.post("/chat")
def chat(body: ChatIn):
    if not client:
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en Render (Environment Variables).")

    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="El mensaje viene vacío.")

    lang = (body.language or "es").lower().strip()
    if lang not in ["es", "en"]:
        lang = "es"

    system_prompt_es = (
        "Eres Corazón Studio AI. Responde con calidez y claridad. "
        "Responde SOLO en español. Sé breve pero útil."
    )
    system_prompt_en = (
        "You are Corazón Studio AI. Respond warmly and clearly. "
        "Respond ONLY in English. Be concise but helpful."
    )
    system_prompt = system_prompt_en if lang == "en" else system_prompt_es

    # =========================
    # Opción A: Assistants (si hay ASSISTANT_ID)
    # =========================
    if ASSISTANT_ID:
        try:
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"[language={lang}] {msg}",
            )

            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=ASSISTANT_ID,
                additional_instructions=system_prompt,
            )

            # Espera simple hasta que termine
            while True:
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run.status in ["completed", "failed", "cancelled", "expired"]:
                    break

            if run.status != "completed":
                raise HTTPException(status_code=500, detail=f"Assistants run terminó con estado: {run.status}")

            msgs = client.beta.threads.messages.list(thread_id=thread.id)
            # Busca el último mensaje del assistant
            for m in msgs.data:
                if m.role == "assistant":
                    text_parts = []
                    for c in m.content:
                        if c.type == "text" and c.text and c.text.value:
                            text_parts.append(c.text.value)
                    answer = "\n".join(text_parts).strip()
                    return {"ok": True, "language": lang, "answer": answer}

            raise HTTPException(status_code=500, detail="No llegó respuesta del assistant.")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error Assistants: {str(e)}")

    # =========================
    # Opción B: Chat Completions (si NO hay ASSISTANT_ID)
    # =========================
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ],
            temperature=0.7,
        )
        answer = resp.choices[0].message.content.strip()
        return {"ok": True, "language": lang, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error OpenAI chat: {str(e)}")
