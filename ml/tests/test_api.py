"""
HTTP/API integration tests for Aoranema Recommendation API.

These tests use:
- the REAL final XGBoost model,
- the REAL final config files,
- the REAL FastAPI app,
- the REAL RecommendationService pipeline.

Run from the `ml/` directory:

    python -m pytest -q tests/test_api.py
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import (
    API_TITLE,
    API_VERSION,
    app,
    create_app,
)
from api.routes.recommendation import (
    get_recommendation_service,
)


# ---------------------------------------------------------------------
# Shared realistic request data
# ---------------------------------------------------------------------

@pytest.fixture()
def movie_catalog():
    return [
        {
            "movie_id": 1,
            "title": "Orbit Protocol",
            "genres": "Action|Sci-Fi|Thriller",
            "tmdb_title": "Orbit Protocol",
            "release_year": 2010,
            "runtime": 125,
            "original_language": "en",
            "director_ids_json": "[1001]",
            "writer_ids_json": "[2001]",
            "top_cast_ids_json": "[3001, 3002]",
            "keyword_ids_json": "[4001, 4002]",
            "production_company_ids_json": "[5001]",
            "collection_ids_json": "[6001]",
            "n_directors": 1,
            "n_writers": 1,
            "n_top_cast": 2,
            "n_keywords": 2,
        },
        {
            # Alias camelCase deliberately tested.
            "movieId": 2,
            "title": "Summer Letter",
            "genres": "Drama|Romance",
            "tmdb_title": "Summer Letter",
            "release_year": 2015,
            "runtime": 108,
            "original_language": "fr",
            "director_ids_json": "[1002]",
            "writer_ids_json": "[2002]",
            "top_cast_ids_json": "[3003, 3004]",
            "keyword_ids_json": "[4003]",
            "production_company_ids_json": "[5002]",
            "collection_ids_json": "[]",
            "n_directors": 1,
            "n_writers": 1,
            "n_top_cast": 2,
            "n_keywords": 1,
        },
        {
            "movie_id": 5,
            "title": "Animated Galaxy",
            "genres": "Adventure|Animation|Children|Sci-Fi",
            "tmdb_title": "Animated Galaxy",
            "release_year": 2024,
            "runtime": 103,
            "original_language": "en",
            "director_ids_json": "[1005]",
            "writer_ids_json": "[2005, 2006]",
            "top_cast_ids_json": "[3010, 3011, 3012]",
            "keyword_ids_json": "[4001, 4011, 4012]",
            "production_company_ids_json": "[5005]",
            "collection_ids_json": "[6005]",
            "n_directors": 1,
            "n_writers": 2,
            "n_top_cast": 3,
            "n_keywords": 3,
        },
    ]


@pytest.fixture()
def candidates():
    return [
        {
            "movie_id": 3,
            "title": "Deep Frontier",
            "genres": "Action|Adventure|Sci-Fi",
            "tmdb_title": "Deep Frontier",
            "release_year": 2022,
            "runtime": 135,
            "original_language": "en",
            "director_ids_json": "[1001]",
            "writer_ids_json": "[2001]",
            "top_cast_ids_json": "[3001, 3005]",
            "keyword_ids_json": "[4001, 4004]",
            "production_company_ids_json": "[5001]",
            "collection_ids_json": "[6001]",
            "n_directors": 1,
            "n_writers": 1,
            "n_top_cast": 2,
            "n_keywords": 2,
            # Laravel/business fields must survive ranking.
            "poster_url": "/poster/deep-frontier.jpg",
            "ticket_price": 50000,
            "cinema_name": "Aoranema Central",
        },
        {
            "movie_id": 4,
            "title": "Laugh Track",
            "genres": "Comedy",
            "tmdb_title": "Laugh Track",
            "release_year": 2021,
            "runtime": 96,
            "original_language": "en",
            "director_ids_json": "[1004]",
            "writer_ids_json": "[2004]",
            "top_cast_ids_json": "[3008, 3009]",
            "keyword_ids_json": "[4010]",
            "production_company_ids_json": "[5004]",
            "collection_ids_json": "[]",
            "n_directors": 1,
            "n_writers": 1,
            "n_top_cast": 2,
            "n_keywords": 1,
            "poster_url": "/poster/laugh-track.jpg",
            "ticket_price": 45000,
            "cinema_name": "Aoranema Central",
        },
    ]


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """
    Start the real app once.

    Entering TestClient triggers FastAPI lifespan, which must preload
    the final recommendation service/model successfully.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """
    Prevent one test's dependency override from leaking into another.
    """
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


def test_health_preloads_real_final_model(client):
    """
    200 health means app lifespan successfully loaded the final model/config.
    """
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert (
        body["service"]["model"]["model_name"]
        == "leakage_safe_full_content"
    )
    assert (
        body["service"]["model"]["feature_count"]
        == 149
    )
    assert (
        body["service"]["model"]["model_feature_count"]
        == 149
    )
    assert (
        body["service"]["default_top_k"]
        == 10
    )
    assert (
        body["service"]["model"]["scores_are_probabilities"]
        is False
    )
    assert (
        body["service"]["model"]["ranking_direction"]
        == "higher_score_is_better"
    )


# ---------------------------------------------------------------------
# Real recommendation flows
# ---------------------------------------------------------------------

def test_history_recommendation_real_model(
    client,
    movie_catalog,
    candidates,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 5.0,
                },
                {
                    "movieId": 2,
                    "rating": 2.5,
                },
            ],
            "movie_catalog": movie_catalog,
            "candidates": candidates,
            "top_k": 2,
            "snapshot_year": 2026,
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["model_name"] == "leakage_safe_full_content"
    assert body["profile_source"] == "history"
    assert body["candidate_count"] == 2
    assert body["returned_count"] == 2
    assert body["warnings"] == []

    recommendations = body["recommendations"]

    assert [
        item["rank"]
        for item in recommendations
    ] == [1, 2]

    scores = np.asarray(
        [
            item["recommendation_score"]
            for item in recommendations
        ],
        dtype=np.float64,
    )

    assert np.isfinite(scores).all()
    assert scores[0] >= scores[1]

    # Candidate IDs preserved, output normalized to snake_case.
    assert sorted(
        item["movie_id"]
        for item in recommendations
    ) == [3, 4]

    assert all(
        "movieId" not in item
        for item in recommendations
    )

    # Laravel/business metadata survives full ML roundtrip.
    assert all(
        "poster_url" in item
        for item in recommendations
    )
    assert all(
        "ticket_price" in item
        for item in recommendations
    )
    assert all(
        "cinema_name" in item
        for item in recommendations
    )


def test_onboarding_recommendation_real_model(
    client,
    movie_catalog,
    candidates,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "onboarding",
            "favorite_movie_ids": [
                1,
                5,
            ],
            "favorite_genres": [
                "Action",
                "Sci-Fi",
                "Adventure",
            ],
            "movie_catalog": movie_catalog,
            "candidates": candidates,
            "top_k": 2,
            "snapshot_year": 2026,
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["profile_source"] == "onboarding"
    assert body["candidate_count"] == 2
    assert body["returned_count"] == 2

    scores = [
        item["recommendation_score"]
        for item in body["recommendations"]
    ]

    assert all(
        np.isfinite(score)
        for score in scores
    )


def test_empty_candidates_returns_empty_success(
    client,
    movie_catalog,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 5.0,
                }
            ],
            "movie_catalog": movie_catalog,
            "candidates": [],
            "top_k": 10,
            "snapshot_year": 2026,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["candidate_count"] == 0
    assert body["returned_count"] == 0
    assert body["recommendations"] == []


# ---------------------------------------------------------------------
# Validation / controlled error behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        # Bad rating.
        {
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 9.0,
                }
            ],
            "movie_catalog": [
                {
                    "movie_id": 1,
                    "genres": "Action",
                }
            ],
            "candidates": [],
        },
        # Unknown mode.
        {
            "mode": "unknown",
            "movie_catalog": [
                {
                    "movie_id": 1,
                    "genres": "Action",
                }
            ],
            "candidates": [],
        },
        # Onboarding with no preference signal.
        {
            "mode": "onboarding",
            "favorite_movie_ids": [],
            "favorite_genres": [],
            "movie_catalog": [
                {
                    "movie_id": 1,
                    "genres": "Action",
                }
            ],
            "candidates": [],
        },
        # Invalid top_k.
        {
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 5.0,
                }
            ],
            "movie_catalog": [
                {
                    "movie_id": 1,
                    "genres": "Action",
                }
            ],
            "candidates": [],
            "top_k": 0,
        },
    ],
)
def test_structurally_invalid_requests_return_422(
    client,
    payload,
):
    response = client.post(
        "/recommendations",
        json=payload,
    )

    assert response.status_code == 422


def test_duplicate_candidate_is_rejected(
    client,
    movie_catalog,
    candidates,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 5.0,
                }
            ],
            "movie_catalog": movie_catalog,
            "candidates": [
                candidates[0],
                candidates[0],
            ],
        },
    )

    assert response.status_code == 422


def test_missing_history_movie_in_catalog_returns_422(
    client,
    movie_catalog,
    candidates,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 999,
                    "rating": 5.0,
                }
            ],
            "movie_catalog": movie_catalog,
            "candidates": candidates,
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert detail["missing_movie_ids"] == [999]


def test_missing_onboarding_movie_in_catalog_returns_422(
    client,
    movie_catalog,
    candidates,
):
    response = client.post(
        "/recommendations",
        json={
            "mode": "onboarding",
            "favorite_movie_ids": [999],
            "favorite_genres": [],
            "movie_catalog": movie_catalog,
            "candidates": candidates,
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert detail["missing_movie_ids"] == [999]


def test_internal_error_is_sanitized(
    client,
    movie_catalog,
    candidates,
):
    """
    Unexpected internals must be logged server-side but not leaked to clients.
    """

    class BrokenService:
        def recommend_from_history(
            self,
            **kwargs,
        ):
            raise RuntimeError(
                "SECRET_INTERNAL_MODEL_PATH"
            )

    def broken_dependency():
        return BrokenService()

    app.dependency_overrides[
        get_recommendation_service
    ] = broken_dependency

    response = client.post(
        "/recommendations",
        json={
            "mode": "history",
            "interactions": [
                {
                    "movie_id": 1,
                    "rating": 5.0,
                }
            ],
            "movie_catalog": movie_catalog,
            "candidates": candidates,
        },
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "Recommendation service failed to process the request."
    )

    assert (
        "SECRET_INTERNAL_MODEL_PATH"
        not in response.text
    )


# ---------------------------------------------------------------------
# App/documentation/deployment behavior
# ---------------------------------------------------------------------

def test_model_service_dependency_is_cached(client):
    first = get_recommendation_service()
    second = get_recommendation_service()

    assert first is second


def test_docs_and_openapi_exist(client):
    docs = client.get("/docs")
    redoc = client.get("/redoc")
    openapi_response = client.get(
        "/openapi.json"
    )

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi_response.status_code == 200

    schema = openapi_response.json()

    assert schema["info"]["title"] == API_TITLE
    assert schema["info"]["version"] == API_VERSION

    assert "/" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/recommendations" in schema["paths"]

    recommendation_operation = (
        schema["paths"]
        ["/recommendations"]
        ["post"]
    )

    assert (
        recommendation_operation["summary"]
        == "Generate personalized movie recommendations"
    )

    assert "200" in (
        recommendation_operation["responses"]
    )
    assert "422" in (
        recommendation_operation["responses"]
    )


def test_unknown_route_and_wrong_method(
    client,
):
    not_found = client.get(
        "/this-route-does-not-exist"
    )

    wrong_method = client.get(
        "/recommendations"
    )

    assert not_found.status_code == 404
    assert wrong_method.status_code == 405


def test_cors_is_opt_in_and_restricted():
    """
    CORS should be enabled only for explicitly configured origins.
    Laravel server-to-server calls do not need browser CORS.
    """
    previous = os.environ.get(
        "AORANEMA_CORS_ORIGINS"
    )

    try:
        os.environ[
            "AORANEMA_CORS_ORIGINS"
        ] = (
            "http://localhost:5173,"
            "http://localhost:8000"
        )

        cors_app = create_app()

        with TestClient(
            cors_app
        ) as cors_client:
            allowed = cors_client.options(
                "/recommendations",
                headers={
                    "Origin": (
                        "http://localhost:5173"
                    ),
                    "Access-Control-Request-Method": (
                        "POST"
                    ),
                    "Access-Control-Request-Headers": (
                        "content-type"
                    ),
                },
            )

            assert (
                allowed.status_code
                == 200
            )

            assert (
                allowed.headers[
                    "access-control-allow-origin"
                ]
                == "http://localhost:5173"
            )

            blocked = cors_client.options(
                "/recommendations",
                headers={
                    "Origin": (
                        "https://evil.example"
                    ),
                    "Access-Control-Request-Method": (
                        "POST"
                    ),
                },
            )

            assert (
                "access-control-allow-origin"
                not in blocked.headers
            )

    finally:
        if previous is None:
            os.environ.pop(
                "AORANEMA_CORS_ORIGINS",
                None,
            )
        else:
            os.environ[
                "AORANEMA_CORS_ORIGINS"
            ] = previous
