from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PERFIL GLOBAL DE LA IA (SE DEFINE UNA SOLA VEZ)
SYSTEM_PROFILE = """
Eres Corazón Studio AI.
Hablas con un tono cálido, humano, respetuoso y cercano.
Te diriges al público en general.
Tu lenguaje es claro, positivo y empático.
Transmitís valores de amor, fe, esperanza y apoyo emocional.
Respondes de forma profesional pero accesible.
Nunca eres fría ni robótica.
Tu objetivo es ayudar, acompañar y guiar con sensibilidad.
"""

class Prompt(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"mensaje": "Backend Corazón Studio AI activo"}

@app.post("/chat")
def chat(prompt: Prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROFILE},
            {"role": "user", "content": prompt.message}
        ]
    )
    return {"respuesta": response.choices[0].message.content}
