from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.sentiment.inference import predict_sentiment


router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment"],
)


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Teks feedback pengguna",
    )


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


@router.post(
    "",
    response_model=SentimentResponse,
)
def analyze_sentiment(
    request: SentimentRequest,
):
    try:
        result = predict_sentiment(
            request.text
        )

        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
        )

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Gagal melakukan analisis sentiment.",
        ) from error