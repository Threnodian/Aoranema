"""
Aoranema - Recommendation API Route
===================================

HTTP layer untuk recommendation service.

Endpoint utama:

    POST /recommendations

Dua mode request:
- history    -> user sudah memiliki explicit rating history
- onboarding -> user baru memilih favorite movies / genres

Route ini sengaja tipis:
- validasi bentuk JSON ditangani Pydantic di api/schemas.py,
- validasi relasi ID request ditangani sebelum masuk pipeline,
- seluruh logic ML tetap berada di src/recommendation_service.py,
- model dimuat sekali lalu di-cache, bukan dimuat ulang tiap request.

Catatan:
`recommendation_score` adalah raw ranking score XGBRanker, bukan probabilitas.
"""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Annotated, Any, Iterable, Set

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.schemas import (
    HistoryRecommendationRequest,
    MovieId,
    OnboardingRecommendationRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from src.recommendation_service import (
    RecommendationService,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    tags=["recommendations"],
)


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    """
    Load and cache the production recommendation service.

    XGBoost model/config harus dimuat sekali per Python process,
    bukan sekali per HTTP request.

    Karena ini FastAPI dependency, test nantinya juga dapat melakukan
    dependency override tanpa mengubah production code.
    """
    return RecommendationService.from_repo()


def _stable_id_list(
    values: Iterable[MovieId],
) -> list[MovieId]:
    """
    Deduplicate IDs while preserving request order.

    Dipakai hanya untuk pesan error yang deterministik.
    """
    result: list[MovieId] = []
    seen: Set[Any] = set()

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def _validate_referenced_movie_ids(
    payload: RecommendationRequest,
) -> None:
    """
    Semantic request validation across fields.

    `api/schemas.py` validates each field individually, while this route
    validates relationships between fields:

    - every history interaction movie must exist in movie_catalog,
    - every onboarding favorite movie must exist in movie_catalog.

    Without this check PreferenceBuilder can intentionally skip unknown
    history items and return a warning. That behavior is useful internally,
    but for the HTTP API a missing catalog record means Laravel sent an
    incomplete request, so returning 422 is safer than silently losing a
    user-preference signal.
    """
    catalog_ids = {
        movie.movie_id
        for movie in payload.movie_catalog
    }

    if isinstance(
        payload,
        HistoryRecommendationRequest,
    ):
        referenced_ids = [
            interaction.movie_id
            for interaction
            in payload.interactions
        ]
        source_name = "interactions"

    elif isinstance(
        payload,
        OnboardingRecommendationRequest,
    ):
        referenced_ids = list(
            payload.favorite_movie_ids
        )
        source_name = "favorite_movie_ids"

    else:
        return

    missing = _stable_id_list(
        movie_id
        for movie_id in referenced_ids
        if movie_id not in catalog_ids
    )

    if missing:
        raise HTTPException(
            status_code=(
                422
            ),
            detail={
                "message": (
                    f"{source_name} mereferensikan movie_id "
                    "yang tidak tersedia di movie_catalog."
                ),
                "missing_movie_ids": missing,
            },
        )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate personalized movie recommendations",
    description=(
        "Ranks eligible movie candidates using the frozen Aoranema "
        "XGBRanker recommendation model. Supports existing users with "
        "rating history and cold-start users from onboarding preferences."
    ),
    response_description=(
        "Ranked recommendation candidates. "
        "Higher recommendation_score means a higher rank; "
        "the score is not a probability."
    ),
)
def recommend_movies(
    payload: RecommendationRequest,
    service: Annotated[
        RecommendationService,
        Depends(
            get_recommendation_service
        ),
    ],
) -> RecommendationResponse:
    """
    Generate Top-K movie recommendations.

    Laravel responsibilities before calling this endpoint:
    - determine which films are eligible/current/upcoming,
    - provide candidate metadata,
    - provide all catalog records referenced by history/onboarding,
    - provide user rating history OR onboarding choices.

    ML API responsibilities:
    - validate request,
    - build user preference profile,
    - build the exact 149 production features,
    - score candidates with the final XGBRanker,
    - return candidates ordered by recommendation score.
    """
    _validate_referenced_movie_ids(
        payload
    )

    try:
        if isinstance(
            payload,
            HistoryRecommendationRequest,
        ):
            result = (
                service.recommend_from_history(
                    interactions=(
                        payload
                        .interactions_for_service()
                    ),
                    movie_catalog=(
                        payload
                        .movie_catalog_for_service()
                    ),
                    candidates=(
                        payload
                        .candidates_for_service()
                    ),
                    top_k=payload.top_k,
                    snapshot_year=(
                        payload.snapshot_year
                    ),
                    min_score=(
                        payload.min_score
                    ),
                )
            )

        elif isinstance(
            payload,
            OnboardingRecommendationRequest,
        ):
            result = (
                service
                .recommend_from_onboarding(
                    favorite_movie_ids=(
                        payload.favorite_movie_ids
                    ),
                    favorite_genres=(
                        payload.favorite_genres
                    ),
                    movie_catalog=(
                        payload
                        .movie_catalog_for_service()
                    ),
                    candidates=(
                        payload
                        .candidates_for_service()
                    ),
                    top_k=payload.top_k,
                    snapshot_year=(
                        payload.snapshot_year
                    ),
                    min_score=(
                        payload.min_score
                    ),
                )
            )

        else:
            # Defensive guard. Discriminated Pydantic union normally makes
            # this branch unreachable.
            raise HTTPException(
                status_code=(
                    422
                ),
                detail=(
                    "Unsupported recommendation request mode."
                ),
            )

        return (
            RecommendationResponse
            .from_service_result(
                result
            )
        )

    except HTTPException:
        raise

    except (
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        # JSON valid, but data is semantically incompatible with the
        # recommendation pipeline.
        logger.info(
            "Recommendation request rejected: %s",
            exc,
        )

        raise HTTPException(
            status_code=(
                422
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Never expose model paths, stack traces, or internal exception text
        # to Laravel/client. Full exception remains in server logs.
        logger.exception(
            "Unexpected recommendation pipeline failure."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Recommendation service failed to process the request."
            ),
        ) from exc


__all__ = [
    "router",
    "get_recommendation_service",
]
