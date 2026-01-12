from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Corazón Studio AI Backend")

# --- CORS (PERMITE QUE GITHUB PAGES SE CONECTE) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELO DE DATOS ---
class ChatRequest(BaseModel):
    message: str

# --- RUTA DE PRUEBA ---
@app.get("/")
def root():
    return {"mensaje": "Backend Corazón Studio AI activo"}

# --- RUTA PRINCIPAL PARA EL FRONTEND ---
@app.post("/chat")
def chat(data: ChatRequest):
    user_message = data.message

    # Respuesta simple (luego aquí va la IA real)
    return {
        "respuesta": f"Recibí tu mensaje: '{user_message}'. Conexión exitosa 💙"
    }
