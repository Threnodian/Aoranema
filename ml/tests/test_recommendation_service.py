"""
Integration tests for Aoranema RecommendationService.

Pipeline covered:

    RecommendationService
        -> PreferenceBuilder
        -> FeatureBuilder
        -> RecommendationPredictor
        -> MovieRanker

Run from the `ml/` directory:

    python -m pytest -q tests/test_recommendation_service.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Make `ml/` importable if pytest is started from another directory.
# tests/test_recommendation_service.py -> parent.parent == ml/
# ---------------------------------------------------------------------
ML_ROOT = Path(__file__).resolve().parents[1]

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


from src.recommendation_service import RecommendationService


# ---------------------------------------------------------------------
# Real production artifacts
# ---------------------------------------------------------------------
MODEL_PATH = (
    ML_ROOT
    / "models"
    / "final_recommender_xgboost_ranker.json"
)

FEATURE_CONFIG_PATH = (
    ML_ROOT
    / "config"
    / "feature_config.json"
)

PREFERENCE_CONFIG_PATH = (
    ML_ROOT
    / "config"
    / "preference_config.json"
)

MODEL_CONFIG_PATH = (
    ML_ROOT
    / "config"
    / "model_config.json"
)


@pytest.fixture(scope="session")
def service():
    """
    Load the actual production RecommendationService once per test session.
    """
    required = [
        MODEL_PATH,
        FEATURE_CONFIG_PATH,
        PREFERENCE_CONFIG_PATH,
        MODEL_CONFIG_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        pytest.fail(
            "Repository artifact belum lengkap. Missing:\n- "
            + "\n- ".join(missing)
        )

    return RecommendationService.from_repo(
        ML_ROOT
    )


@pytest.fixture()
def movie_catalog():
    """
    Synthetic but structurally realistic metadata catalog.
    """
    return {
        1: {
            "movieId": 1,
            "title": "Orbit Protocol (2010)",
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
        2: {
            "movieId": 2,
            "title": "Summer Letter (2015)",
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
        3: {
            "movieId": 3,
            "title": "Deep Frontier (2022)",
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
        },
        4: {
            "movieId": 4,
            "title": "Laugh Track (2021)",
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
        },
        5: {
            "movieId": 5,
            "title": "Animated Galaxy (2024)",
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
    }


def test_service_contract(service):
    """
    Service must load the frozen final model/config consistently.
    """
    info = service.info()

    assert info["service"] == "aoranema_movie_recommendation"
    assert info["model"]["model_name"] == "leakage_safe_full_content"
    assert info["model"]["feature_count"] == 149
    assert info["model"]["model_feature_count"] == 149
    assert info["default_top_k"] == 10
    assert info["pipeline"] == [
        "preference_builder",
        "feature_builder",
        "predictor",
        "ranker",
    ]


def test_history_flow_end_to_end(
    service,
    movie_catalog,
):
    """
    User with explicit rating history should receive deterministic ranked output.
    """
    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[5],
    ]

    result = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
        candidates=candidates,
        top_k=3,
        snapshot_year=2026,
    )

    assert result.model_name == "leakage_safe_full_content"
    assert result.profile_source == "history"
    assert result.candidate_count == 3
    assert result.returned_count == 3

    assert [
        item["rank"]
        for item in result.recommendations
    ] == [1, 2, 3]

    scores = np.asarray(
        [
            item["recommendation_score"]
            for item in result.recommendations
        ],
        dtype=np.float32,
    )

    assert np.isfinite(scores).all()
    assert np.all(
        scores[:-1] >= scores[1:]
    )

    returned_ids = [
        item["movieId"]
        for item in result.recommendations
    ]

    assert sorted(returned_ids) == [3, 4, 5]


def test_onboarding_flow_end_to_end(
    service,
    movie_catalog,
):
    """
    Cold-start user should work using favorite movies + favorite genres.
    """
    result = service.recommend_from_onboarding(
        favorite_movie_ids=[1, 5],
        favorite_genres=[
            "Action",
            "Sci-Fi",
            "Adventure",
        ],
        movie_catalog=movie_catalog,
        candidates=[
            movie_catalog[3],
            movie_catalog[4],
            movie_catalog[5],
        ],
        top_k=2,
        snapshot_year=2026,
    )

    assert result.profile_source == "onboarding"
    assert result.candidate_count == 3
    assert result.returned_count == 2

    assert [
        item["rank"]
        for item in result.recommendations
    ] == [1, 2]

    assert np.isfinite(
        [
            item["recommendation_score"]
            for item in result.recommendations
        ]
    ).all()


def test_service_result_matches_manual_pipeline(
    service,
    movie_catalog,
):
    """
    The service must only orchestrate components, not silently alter their logic.
    """
    interactions = [
        {"movie_id": 1, "rating": 5.0},
        {"movie_id": 2, "rating": 2.5},
    ]

    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[5],
    ]

    service_result = service.recommend_from_history(
        interactions=interactions,
        movie_catalog=movie_catalog,
        candidates=candidates,
        top_k=3,
        snapshot_year=2026,
    )

    profile = (
        service.preference_builder
        .build_profile(
            interactions=interactions,
            movie_catalog=movie_catalog,
            source="history",
        )
    )

    feature_batch = (
        service.feature_builder
        .build_matrix(
            profile=profile,
            candidates=candidates,
            snapshot_year=2026,
        )
    )

    prediction = (
        service.predictor
        .predict_feature_batch(
            feature_batch
        )
    )

    manual_result = (
        service.ranker
        .rank_from_feature_batch(
            candidates=candidates,
            scores=prediction.scores,
            feature_batch=feature_batch,
            top_k=3,
        )
    )

    assert (
        service_result.recommendations
        == manual_result
    )


def test_empty_candidate_list(
    service,
    movie_catalog,
):
    """
    Empty availability from Laravel should safely return an empty recommendation.
    """
    result = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
        ],
        movie_catalog=movie_catalog,
        candidates=[],
        snapshot_year=2026,
    )

    assert result.recommendations == []
    assert result.candidate_count == 0
    assert result.returned_count == 0


def test_top_k_greater_than_candidate_count(
    service,
    movie_catalog,
):
    """
    Top-10 request with only two available movies should return two, not fail.
    """
    result = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
        ],
        movie_catalog=movie_catalog,
        candidates=[
            movie_catalog[3],
            movie_catalog[4],
        ],
        top_k=10,
        snapshot_year=2026,
    )

    assert result.candidate_count == 2
    assert result.returned_count == 2
    assert len(result.recommendations) == 2


def test_candidate_input_is_not_mutated(
    service,
    movie_catalog,
):
    """
    The service should never add rank/score directly into caller-owned objects.
    """
    candidates = [
        movie_catalog[3].copy(),
        movie_catalog[4].copy(),
        movie_catalog[5].copy(),
    ]

    original = [
        item.copy()
        for item in candidates
    ]

    _ = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
        candidates=candidates,
        top_k=3,
        snapshot_year=2026,
    )

    assert candidates == original


def test_generator_candidates_are_supported(
    service,
    movie_catalog,
):
    """
    Candidate iterable may come from DB/API generators and must be consumed safely.
    """
    def generate_candidates():
        yield movie_catalog[3]
        yield movie_catalog[4]
        yield movie_catalog[5]

    result = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
        candidates=generate_candidates(),
        top_k=3,
        snapshot_year=2026,
    )

    assert result.candidate_count == 3
    assert result.returned_count == 3


def test_min_score_filter_is_respected(
    service,
    movie_catalog,
):
    """
    min_score is optional because XGBRanker score is not probability.
    If caller uses it, service must pass it through correctly.
    """
    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[5],
    ]

    baseline = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
        candidates=candidates,
        top_k=3,
        snapshot_year=2026,
    )

    top_score = float(
        baseline.recommendations[0][
            "recommendation_score"
        ]
    )

    filtered = service.recommend_from_history(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
        candidates=candidates,
        top_k=3,
        snapshot_year=2026,
        min_score=top_score,
    )

    assert filtered.returned_count >= 1

    for item in filtered.recommendations:
        assert (
            item["recommendation_score"]
            >= top_score
        )
