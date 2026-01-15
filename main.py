from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI(title="Corazón Studio AI API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------
# MODELOS
# ---------

class ChatRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    prompt: str

# ----------------
# ENDPOINTS BASE
# ----------------

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ----------------
# CHAT (SIMULADO)
# ----------------

@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "response": f"Respuesta simulada para: {req.prompt}"
    }

# ----------------
# IMAGE (SIMULADO)
# ----------------

@app.post("/image")
def image(req: ImageRequest):
    return {
        "image_url": "https://placehold.co/512x512?text=Imagen+Generada"
    }

# ----------------
# VIDEO OPCIÓN 1
# (REELS SIMULADO)
# ----------------

VIDEOS = {}

@app.post("/video")
def start_video(req: VideoRequest):
    video_id = f"video_{uuid.uuid4().hex[:12]}"
    VIDEOS[video_id] = {
        "status": "completed",
        "url": "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
    }
    return {
        "video_id": video_id,
        "status": "completed"
    }

@app.get("/video/{video_id}")
def video_status(video_id: str):
    if video_id not in VIDEOS:
        raise HTTPException(status_code=404, detail="video_id no existe")
    return {
        "video_id": video_id,
        "status": VIDEOS[video_id]["status"]
    }

@app.get("/video/{video_id}/content")
def video_content(video_id: str):
    if video_id not in VIDEOS:
        raise HTTPException(status_code=404, detail="video_id no existe")
    return {
        "video_url": VIDEOS[video_id]["url"]
    }
