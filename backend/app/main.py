from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import query, health

app = FastAPI(title="Decision Intelligence Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)

@app.get("/")
async def root():
    return {"message": "Decision Intelligence Assistant API"}