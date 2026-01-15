from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import uuid
import time

app = FastAPI()

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MEMORIA TEMPORAL
# ===============================
jobs = {}

# ===============================
# CHAT
# ===============================
@app.post("/chat")
def chat(data: dict):
    message = data.get("message", "")
    return {
        "reply": f"💙 Mensaje recibido: {message}"
    }

# ===============================
# IMÁGENES
# ===============================
@app.post("/generate-image")
def generate_image(data: dict):
    prompt = data.get("prompt", "")
    return {
        "image_url": "https://via.placeholder.com/512x512.png?text=Imagen+Generada"
    }

# ===============================
# VIDEO (BACKGROUND - Render FREE)
# ===============================
def generate_video_background(job_id: str, prompt: str):
    # Simula proceso largo
    time.sleep(20)

    jobs[job_id]["status"] = "done"
    jobs[job_id]["video_url"] = "https://www.w3schools.com/html/mov_bbb.mp4"

@app.post("/generate-video")
def generate_video(data: dict):
    prompt = data.get("prompt", "")
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing"
    }

    thread = threading.Thread(
        target=generate_video_background,
        args=(job_id, prompt)
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "processing"
    }

@app.get("/video-status/{job_id}")
def video_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

# ===============================
# ROOT
# ===============================
@app.get("/")
def root():
    return {"status": "Corazón Studio AI backend activo"}
