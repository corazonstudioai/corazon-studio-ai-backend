from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,

    allow_origins=["https://corazon-studio-ai-web.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return{"mensaje": "Backend Corazón Studio AI activo"}
       
           class DemoVideoRequest(BaseModel):
           prompt: str


@app.post("/demo-video")
def demo_video(data:
DemoVideoRequest):
return{
"status": "ok",
"mensaje": "Video demo en proceso",
"prompt_recibido":data.prompt,"tipo":"demo",
}
