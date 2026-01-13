from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

# ---------- APP ----------
app = FastAPI(title="Corazón Studio AI Backend")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OPENAI ----------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------- MODELO ----------
class ChatRequest(BaseModel):
    message: str

# ---------- RUTAS ----------
@app.get("/")
def home():
    return {"mensaje": "Backend Corazón Studio AI activo ❤️"}

@app.post("/chat")
def chat(data: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres una IA con propósito, empática y clara."},
                {"role": "user", "content": data.message}
            ]
        )

        return {
            "respuesta": response.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}
