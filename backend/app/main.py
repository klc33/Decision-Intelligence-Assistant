from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import query, health  # ← changed

app = FastAPI(title="Decision Intelligence Assistant API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)

@app.get("/")
async def root():
    return {"message": "Decision Intelligence Assistant API"}