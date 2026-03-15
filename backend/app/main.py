# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import prediction,fetch_data, sentiment
from backend.app.database.db_setup import Base, engine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sentiment Stock Predictor API")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)
app.include_router(fetch_data.router)
app.include_router(sentiment.router)

@app.get("/")
def root():
    return {"message": "✅ Sentiment Stock Predictor API running."}
