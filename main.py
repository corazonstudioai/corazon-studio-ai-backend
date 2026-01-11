from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            {"role": "system", "content": "Eres una IA con propósito, amable y clara."},
            {"role": "user", "content": prompt.message},
        ],
    )
    return {"respuesta": response.choices[0].message.content}
