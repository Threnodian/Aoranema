"""
End-to-end tests for Aoranema recommendation pipeline.

Pipeline under test:

    PreferenceBuilder
        -> FeatureBuilder
        -> RecommendationPredictor
        -> MovieRanker
        -> Top-N recommendations

Run from `ml/`:

    pytest -q

or only this file:

    pytest -q tests/test_recommendation_pipeline.py

The tests intentionally use a small synthetic movie catalog, but they load
the REAL final model and REAL config files from the repository.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Make `ml/` importable when pytest is started from another directory.
# tests/test_recommendation_pipeline.py -> parent.parent == ml/
# ---------------------------------------------------------------------
ML_ROOT = Path(__file__).resolve().parents[1]

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


from src.preference_builder import PreferenceBuilder
from src.feature_builder import FeatureBuilder
from src.predictor import RecommendationPredictor
from src.ranker import MovieRanker


# ---------------------------------------------------------------------
# Repository artifacts expected by the production pipeline.
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
def pipeline():
    """
    Load the exact production components once for the whole test session.
    """
    required_paths = [
        MODEL_PATH,
        FEATURE_CONFIG_PATH,
        PREFERENCE_CONFIG_PATH,
        MODEL_CONFIG_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        pytest.fail(
            "Repository artifact belum lengkap. Missing:\n- "
            + "\n- ".join(missing)
        )

    preference_builder = (
        PreferenceBuilder.from_config_files(
            FEATURE_CONFIG_PATH,
            PREFERENCE_CONFIG_PATH,
        )
    )

    feature_builder = (
        FeatureBuilder.from_config_files(
            FEATURE_CONFIG_PATH,
            MODEL_CONFIG_PATH,
        )
    )

    predictor = (
        RecommendationPredictor.from_files(
            model_path=MODEL_PATH,
            feature_config_path=FEATURE_CONFIG_PATH,
            model_config_path=MODEL_CONFIG_PATH,
        )
    )

    ranker = MovieRanker.from_model_config(
        MODEL_CONFIG_PATH
    )

    return {
        "preference_builder": preference_builder,
        "feature_builder": feature_builder,
        "predictor": predictor,
        "ranker": ranker,
    }


@pytest.fixture()
def movie_catalog():
    """
    Small deterministic catalog.

    Metadata fields intentionally follow the same shapes used by the
    production builders:
    - MovieLens-style genre names
    - TMDB entity IDs serialized as JSON strings
    - release year / runtime / language
    - metadata counts
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
        6: {
            "movieId": 6,
            "title": "Unknown Runtime Candidate",
            # TMDB-style genre names deliberately test production mapping.
            "tmdb_genre_names_json": '["Science Fiction", "Family"]',
            "tmdb_title": "Unknown Runtime Candidate",
            "release_date": "2026-07-10",
            "runtime": 0,
            "original_language": "ja",
            "director_ids_json": "[1001]",
            "writer_ids_json": "[2001]",
            "top_cast_ids_json": "[3001, 3020]",
            "keyword_ids_json": "[4001]",
            "production_company_ids_json": "[5001]",
            "collection_ids_json": "[]",
            "n_directors": 1,
            "n_writers": 1,
            "n_top_cast": 2,
            "n_keywords": 1,
        },
    }


def test_final_model_contract(pipeline):
    """
    Production config and model must still agree with the frozen final audit.
    """
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]
    ranker = pipeline["ranker"]

    assert predictor.model_name == "leakage_safe_full_content"
    assert predictor.expected_feature_count == 149
    assert predictor.model_feature_count == 149
    assert len(feature_builder.final_feature_names) == 149
    assert ranker.default_top_k == 10

    # Features removed by the final leakage audit must never return.
    assert (
        "movie_historical_rating_norm"
        not in feature_builder.final_feature_names
    )
    assert (
        "movie_historical_log_count_norm"
        not in feature_builder.final_feature_names
    )


def test_full_recommendation_pipeline(
    pipeline,
    movie_catalog,
):
    """
    Main end-to-end test:
    history -> profile -> features -> scores -> ranking.
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]
    ranker = pipeline["ranker"]

    profile = preference_builder.build_profile(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.5},
        ],
        movie_catalog=movie_catalog,
    )

    assert profile.history_count == 2
    assert np.isfinite(profile.user_mean_rating)

    # In real production these are films Laravel has already filtered as
    # currently available/upcoming candidates.
    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[5],
        movie_catalog[6],
    ]

    feature_batch = feature_builder.build_matrix(
        profile=profile,
        candidates=candidates,
        snapshot_year=2026,
    )

    assert feature_batch.matrix.shape == (4, 149)
    assert feature_batch.matrix.dtype == np.float32
    assert np.isfinite(feature_batch.matrix).all()
    assert feature_batch.movie_ids == [3, 4, 5, 6]

    prediction = predictor.predict_feature_batch(
        feature_batch
    )

    assert prediction.scores.shape == (4,)
    assert prediction.movie_ids == [3, 4, 5, 6]
    assert np.isfinite(prediction.scores).all()

    ranked = ranker.rank_from_feature_batch(
        candidates=candidates,
        scores=prediction.scores,
        feature_batch=feature_batch,
        top_k=3,
    )

    assert len(ranked) == 3
    assert [row["rank"] for row in ranked] == [1, 2, 3]

    ranked_scores = np.asarray(
        [
            row["recommendation_score"]
            for row in ranked
        ],
        dtype=np.float32,
    )

    # Descending ranking.
    assert np.all(
        ranked_scores[:-1]
        >= ranked_scores[1:]
    )

    # Independent order check.
    expected_order = np.argsort(
        -prediction.scores,
        kind="stable",
    )[:3]

    expected_ids = [
        candidates[int(index)]["movieId"]
        for index in expected_order
    ]

    actual_ids = [
        row["movieId"]
        for row in ranked
    ]

    assert actual_ids == expected_ids


def test_onboarding_cold_start_pipeline(
    pipeline,
    movie_catalog,
):
    """
    A new Aoranema user with no real ratings can still get a profile
    from onboarding liked movies + favorite genres.
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]
    ranker = pipeline["ranker"]

    profile = (
        preference_builder.build_onboarding_profile(
            favorite_movie_ids=[1, 5],
            favorite_genres=[
                "Action",
                "Sci-Fi",
                "Adventure",
            ],
            movie_catalog=movie_catalog,
        )
    )

    assert profile.source == "onboarding"
    assert profile.history_count == 2

    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[6],
    ]

    feature_batch = feature_builder.build_matrix(
        profile,
        candidates,
        snapshot_year=2026,
    )

    prediction = predictor.predict_feature_batch(
        feature_batch
    )

    ranked = ranker.rank_from_feature_batch(
        candidates=candidates,
        scores=prediction.scores,
        feature_batch=feature_batch,
        top_k=3,
    )

    assert len(ranked) == 3
    assert np.isfinite(
        [
            row["recommendation_score"]
            for row in ranked
        ]
    ).all()


def test_top_k_larger_than_candidate_count(
    pipeline,
    movie_catalog,
):
    """
    Asking Top-10 with only two candidates should simply return two.
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]
    ranker = pipeline["ranker"]

    profile = preference_builder.build_profile(
        interactions=[
            {"movie_id": 1, "rating": 4.5},
        ],
        movie_catalog=movie_catalog,
    )

    candidates = [
        movie_catalog[3],
        movie_catalog[4],
    ]

    feature_batch = feature_builder.build_matrix(
        profile,
        candidates,
        snapshot_year=2026,
    )

    scores = predictor.predict_feature_batch(
        feature_batch
    ).scores

    ranked = ranker.rank_from_feature_batch(
        candidates=candidates,
        scores=scores,
        feature_batch=feature_batch,
        top_k=10,
    )

    assert len(ranked) == 2


def test_ranker_rejects_candidate_order_mismatch(
    pipeline,
    movie_catalog,
):
    """
    Prevent the dangerous bug:
        features for Film A
        but score accidentally attached to Film B.
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]
    ranker = pipeline["ranker"]

    profile = preference_builder.build_profile(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
            {"movie_id": 2, "rating": 2.0},
        ],
        movie_catalog=movie_catalog,
    )

    candidates = [
        movie_catalog[3],
        movie_catalog[4],
        movie_catalog[5],
    ]

    feature_batch = feature_builder.build_matrix(
        profile,
        candidates,
        snapshot_year=2026,
    )

    scores = predictor.predict_feature_batch(
        feature_batch
    ).scores

    wrong_order = [
        candidates[1],
        candidates[0],
        candidates[2],
    ]

    with pytest.raises(
        ValueError,
        match="Urutan/ID candidate",
    ):
        ranker.rank_from_feature_batch(
            candidates=wrong_order,
            scores=scores,
            feature_batch=feature_batch,
            top_k=3,
        )


def test_predictor_rejects_wrong_feature_order(
    pipeline,
    movie_catalog,
):
    """
    Predictor must reject a FeatureBatch whose feature-name order changes.
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]
    predictor = pipeline["predictor"]

    profile = preference_builder.build_profile(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
        ],
        movie_catalog=movie_catalog,
    )

    batch = feature_builder.build_matrix(
        profile,
        [movie_catalog[3]],
        snapshot_year=2026,
    )

    # Simple test double preserving matrix/movie_ids but corrupting order.
    class BadBatch:
        pass

    bad = BadBatch()
    bad.matrix = batch.matrix
    bad.movie_ids = batch.movie_ids
    bad.feature_names = list(
        reversed(
            batch.feature_names
        )
    )

    with pytest.raises(
        ValueError,
        match="Urutan/nama feature",
    ):
        predictor.predict_feature_batch(
            bad
        )


def test_production_metadata_fallbacks(
    pipeline,
    movie_catalog,
):
    """
    Cross-check production-specific preprocessing:
    - TMDB Science Fiction -> MovieLens Sci-Fi
    - TMDB Family -> MovieLens Children
    - runtime=0 -> missing
    """
    preference_builder = pipeline["preference_builder"]
    feature_builder = pipeline["feature_builder"]

    profile = preference_builder.build_profile(
        interactions=[
            {"movie_id": 1, "rating": 5.0},
        ],
        movie_catalog=movie_catalog,
    )

    features = feature_builder.build_feature_dict(
        profile,
        movie_catalog[6],
        snapshot_year=2026,
    )

    assert features["movie_genre_sci_fi"] == 1.0
    assert features["movie_genre_children"] == 1.0
    assert features["movie_has_runtime"] == 0.0

    # Missing runtime is filled by training runtime center,
    # therefore scaled value is exactly zero.
    assert features["movie_runtime_scaled"] == pytest.approx(
        0.0,
        abs=1e-7,
    )
