import os
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

_device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(_device)
    model.eval()
    return model, tokenizer


def predict(text: str):
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(_device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id].item())

    all_scores = {
        model.config.id2label[i]: float(probs[i].item())
        for i in range(probs.shape[-1])
    }

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in all_scores.items()},
    }
