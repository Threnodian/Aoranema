from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from .preprocessing import clean_text


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LENGTH = 128

# inference.py berada di:
# ml/src/sentiment/inference.py
#
# parents[2] = folder ml
ML_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ML_ROOT
    / "models"
    / "sentiment"
    / "indobert_sentiment_model"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model sentiment tidak ditemukan di: {MODEL_PATH}"
    )


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


# ============================================================
# INFERENCE
# ============================================================

def predict_sentiment(text: str) -> dict:
    """
    Prediksi sentiment dari satu teks.

    Output:
    {
        "sentiment": "negative",
        "confidence": 0.98
    }
    """

    text = clean_text(text)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    prediction_id = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[prediction_id].item()
    )

    sentiment = model.config.id2label[
        prediction_id
    ]

    return {
        "sentiment": sentiment,
        "confidence": confidence,
    }


# ============================================================
# OPTIONAL: DETAIL PROBABILITY
# ============================================================

def predict_sentiment_detail(text: str) -> dict:
    """
    Versi lengkap untuk debugging / eksperimen.

    Menampilkan probabilitas seluruh kelas.
    """

    text = clean_text(text)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    prediction_id = int(
        torch.argmax(probabilities).item()
    )

    sentiment = model.config.id2label[
        prediction_id
    ]

    probability_by_label = {}

    for label_id, probability in enumerate(probabilities):
        label = model.config.id2label[
            label_id
        ]

        probability_by_label[label] = float(
            probability.item()
        )

    return {
        "sentiment": sentiment,
        "confidence": float(
            probabilities[prediction_id].item()
        ),
        "probabilities": probability_by_label,
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":
    examples = [
        "Proses booking tiketnya cepat dan gampang banget.",
        "Pembayaran QRIS sering gagal dan bikin kesal.",
        "Aplikasinya biasa saja.",
        "Customer service sangat ramah dan membantu.",
    ]

    print("Device :", DEVICE)
    print("Model  :", MODEL_PATH)
    print()

    for text in examples:
        result = predict_sentiment_detail(text)

        print("=" * 60)
        print("Text       :", text)
        print("Sentiment  :", result["sentiment"])
        print(
            "Confidence :",
            round(result["confidence"], 4),
        )
        print(
            "Probability:",
            {
                key: round(value, 4)
                for key, value
                in result["probabilities"].items()
            },
        )