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
    allow_origins=["*"],  # para pruebas está bien
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# "DB" temporal en memoria
# ===============================
video_jobs = {}

# ===============================
# HEALTH
# ===============================
@app.get("/")
def root():
    return {"status": "Corazón Studio AI backend activo"}

# ===============================
# CHAT
# ===============================
@app.post("/chat")
def chat(data: dict):
    msg = (data.get("message") or "").strip()
    if not msg:
        return {"reply": "Escribe un mensaje para poder responder 😊"}
    return {"reply": f"💬 Recibí: {msg}"}

# ===============================
# IMÁGENES
# ===============================
@app.post("/generate-image")
def generate_image(data: dict):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return {"image_url": "", "error": "Prompt vacío"}
    return {"image_url": "https://via.placeholder.com/768x768.png?text=Imagen+Generada"}

# ===============================
# VIDEO (Render FREE friendly)
# ===============================
def _video_worker(job_id: str, prompt: str):
    time.sleep(20)  # simula proceso largo
    video_jobs[job_id]["status"] = "done"
    video_jobs[job_id]["video_url"] = "https://www.w3schools.com/html/mov_bbb.mp4"

@app.post("/generate-video")
def generate_video(data: dict):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return {"error": "Prompt vacío"}

    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing"}

    t = threading.Thread(target=_video_worker, args=(job_id, prompt), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "processing"}

@app.get("/video-status/{job_id}")
def video_status(job_id: str):
    return video_jobs.get(job_id, {"status": "not_found"})
