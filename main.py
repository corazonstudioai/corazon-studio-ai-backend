from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
app = FastAPI() 
app.add_middleware( CORSMiddleware,
                   allow_origins=["https://corazon-studio-ai-web.onrender.com"], 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"], 
                  )
@app.get("/") def read_root(): return {"mensaje": "Backend Corazón Studio AI activo"}
