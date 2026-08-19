from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model import predict, get_model_and_tokenizer, MODEL_NAME

app = FastAPI(title="Sentiment Prediction Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict


@app.on_event("startup")
def load_model_on_startup():
    get_model_and_tokenizer()
    print(f"Model loaded: {MODEL_NAME}")


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return predict(request.text)
