"""
Aoranema - FastAPI Application
==============================

Entry point untuk Recommendation API.

Jalankan dari folder `ml/`:

    uvicorn api.main:app --reload

atau:

    python -m uvicorn api.main:app --reload

Endpoint utama:
- GET  /               -> info singkat API
- GET  /health         -> readiness/model health
- POST /recommendations -> personalized movie ranking
- GET  /docs           -> Swagger UI
- GET  /redoc          -> ReDoc

Environment variable opsional:
- AORANEMA_CORS_ORIGINS
  Daftar origin dipisahkan koma, contoh:

      AORANEMA_CORS_ORIGINS=http://localhost:5173,http://localhost:8000

Jika kosong/tidak diset, CORS middleware tidak diaktifkan.
Laravel server-to-server tidak membutuhkan CORS.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import AsyncIterator, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes.recommendation import (
    get_recommendation_service,
    router as recommendation_router,
)
from api.schemas import (
    HealthResponse,
    ServiceInfoResponse,
)


API_TITLE = "Aoranema Recommendation API"
API_DESCRIPTION = (
    "Machine-learning recommendation service for Aoranema. "
    "The API builds user preferences, constructs the frozen 149-feature "
    "representation, scores eligible movie candidates with the final "
    "XGBRanker model, and returns ranked recommendations."
)
API_VERSION = "1.0.0"


def _parse_cors_origins(
    raw_value: str | None,
) -> List[str]:
    """
    Parse comma-separated CORS origins.

    - whitespace is trimmed,
    - empty entries are ignored,
    - duplicate origins are removed while preserving order.
    """
    if raw_value is None:
        return []

    result: List[str] = []
    seen = set()

    for part in raw_value.split(","):
        origin = part.strip()

        if not origin:
            continue

        if origin in seen:
            continue

        seen.add(origin)
        result.append(origin)

    return result


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Preload the recommendation service once at application startup.

    This is deliberate fail-fast behavior:
    if model/config artifacts are missing or incompatible, the API should
    fail during startup rather than return unpredictable errors later.

    The route dependency itself is lru-cached, so this does not create a
    second model instance.
    """
    service = get_recommendation_service()

    app.state.recommendation_service = service
    app.state.model_name = service.predictor.model_name
    app.state.model_feature_count = (
        service.predictor.model_feature_count
    )

    yield


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Keeping an app factory makes local tests and future deployment config
    easier without duplicating application setup.
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Large recommendation responses may contain movie metadata/poster/schedule
    # fields. Compress responses >= 1 KB without changing API semantics.
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1024,
    )

    cors_origins = _parse_cors_origins(
        os.getenv(
            "AORANEMA_CORS_ORIGINS"
        )
    )

    if cors_origins:
        wildcard = "*" in cors_origins

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            # Keep credentials disabled by default. This avoids the insecure
            # wildcard+credentials combination and is enough for ordinary
            # token/header-based development calls.
            allow_credentials=False,
            allow_methods=[
                "GET",
                "POST",
                "OPTIONS",
            ],
            allow_headers=[
                "Accept",
                "Authorization",
                "Content-Type",
            ],
            expose_headers=[],
        )

    app.include_router(
        recommendation_router
    )

    @app.get(
        "/",
        summary="API information",
        tags=["system"],
    )
    def root() -> Dict[str, str]:
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "status": "ok",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Recommendation service health",
        tags=["system"],
    )
    def health() -> HealthResponse:
        """
        Readiness endpoint.

        A 200 response means the application started successfully with the
        production recommendation model/config contract loaded.
        """
        service = (
            app.state.recommendation_service
        )

        service_info = (
            ServiceInfoResponse
            .from_service_info(
                service.info()
            )
        )

        return HealthResponse(
            status="ok",
            service=service_info,
        )

    return app


app = create_app()


__all__ = [
    "app",
    "create_app",
]
