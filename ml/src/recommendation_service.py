"""
Aoranema - Recommendation Service
=================================

Orchestrator untuk seluruh pipeline recommendation production:

    histori / onboarding user
        ↓
    PreferenceBuilder
        ↓
    FeatureBuilder
        ↓
    RecommendationPredictor
        ↓
    MovieRanker
        ↓
    Top-N recommendation

File ini TIDAK melatih model dan TIDAK mengubah bobot preference berdasarkan
wishlist/booking. Logic event seperti wishlist/booking nanti ditangani terpisah
oleh preference_updater.py setelah alur recommendation utama stabil.

Service ini dibuat agar FastAPI nantinya cukup memanggil satu object, bukan
memanggil empat komponen ML satu per satu.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .preference_builder import (
    Interaction,
    PreferenceBuilder,
    UserPreferenceProfile,
)
from .feature_builder import FeatureBuilder
from .predictor import RecommendationPredictor
from .ranker import MovieRanker


MovieRecord = Mapping[str, Any]
MovieId = Union[int, str]


@dataclass
class RecommendationResult:
    """
    Hasil lengkap satu request recommendation.

    recommendations:
        Candidate yang sudah diurutkan dari score tertinggi ke terendah.

    candidate_count:
        Jumlah candidate yang benar-benar dikirim ke model.

    returned_count:
        Jumlah film yang dikembalikan setelah Top-K / min_score.

    model_name:
        Model final yang dipakai.

    profile_source:
        "history", "onboarding", atau label lain dari UserPreferenceProfile.

    warnings:
        Warning non-fatal dari proses pembuatan preference.
    """

    recommendations: List[Dict[str, Any]]
    candidate_count: int
    returned_count: int
    model_name: str
    profile_source: str
    warnings: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecommendationService:
    """
    Facade/orchestrator production recommendation Aoranema.

    Contoh history user:

        service = RecommendationService.from_repo()

        result = service.recommend_from_history(
            interactions=[
                {"movie_id": 1, "rating": 5.0},
                {"movie_id": 2, "rating": 3.0},
            ],
            movie_catalog=movie_catalog,
            candidates=movies_now_playing,
            top_k=10,
        )

    Contoh user baru / onboarding:

        result = service.recommend_from_onboarding(
            favorite_movie_ids=[1, 50, 260],
            favorite_genres=["Action", "Sci-Fi"],
            movie_catalog=movie_catalog,
            candidates=movies_now_playing,
            top_k=10,
        )
    """

    def __init__(
        self,
        *,
        preference_builder: PreferenceBuilder,
        feature_builder: FeatureBuilder,
        predictor: RecommendationPredictor,
        ranker: MovieRanker,
    ) -> None:
        self.preference_builder = preference_builder
        self.feature_builder = feature_builder
        self.predictor = predictor
        self.ranker = ranker

        self._validate_component_contract()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_repo(
        cls,
        ml_root: Optional[Union[str, Path]] = None,
    ) -> "RecommendationService":
        """
        Membaca seluruh config/model dari struktur repository:

            ml/
            ├── config/
            ├── models/
            └── src/
        """
        if ml_root is None:
            ml_root = (
                Path(__file__)
                .resolve()
                .parents[1]
            )
        else:
            ml_root = Path(ml_root)

        return cls(
            preference_builder=(
                PreferenceBuilder.from_repo_config(
                    ml_root
                )
            ),
            feature_builder=(
                FeatureBuilder.from_repo_config(
                    ml_root
                )
            ),
            predictor=(
                RecommendationPredictor.from_repo(
                    ml_root
                )
            ),
            ranker=(
                MovieRanker.from_repo_config(
                    ml_root
                )
            ),
        )

    # Alias agar nama constructor konsisten dengan class lain.
    from_repo_config = from_repo

    # ------------------------------------------------------------------
    # Main public flows
    # ------------------------------------------------------------------

    def recommend_from_history(
        self,
        *,
        interactions: Iterable[
            Union[
                Interaction,
                Mapping[str, Any],
            ]
        ],
        movie_catalog: Union[
            Mapping[MovieId, MovieRecord],
            Iterable[MovieRecord],
        ],
        candidates: Iterable[MovieRecord],
        top_k: Optional[int] = None,
        snapshot_year: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RecommendationResult:
        """
        Recommendation untuk user yang sudah punya explicit rating/history.

        Catatan:
        Service tidak otomatis menghapus candidate yang sudah pernah dirating.
        Pemilihan film yang memang tersedia/layak direkomendasikan tetap menjadi
        tanggung jawab layer business/backend (Laravel) sebelum request dikirim.
        """
        profile = self.preference_builder.build_profile(
            interactions=interactions,
            movie_catalog=movie_catalog,
            source="history",
        )

        return self.recommend_with_profile(
            profile=profile,
            candidates=candidates,
            top_k=top_k,
            snapshot_year=snapshot_year,
            min_score=min_score,
        )

    def recommend_from_onboarding(
        self,
        *,
        favorite_movie_ids: Sequence[MovieId],
        favorite_genres: Sequence[str],
        movie_catalog: Union[
            Mapping[MovieId, MovieRecord],
            Iterable[MovieRecord],
        ],
        candidates: Iterable[MovieRecord],
        top_k: Optional[int] = None,
        snapshot_year: Optional[int] = None,
        min_score: Optional[float] = None,
        liked_movie_pseudo_rating: float = 4.5,
        favorite_genre_pseudo_rating: float = 4.5,
    ) -> RecommendationResult:
        """
        Recommendation untuk user baru.

        liked movie + favorite genre hanya dipakai untuk membentuk profile awal.
        Ini cold-start strategy yang sudah didukung preference_builder.py.
        """
        profile = (
            self.preference_builder
            .build_onboarding_profile(
                favorite_movie_ids=(
                    favorite_movie_ids
                ),
                favorite_genres=(
                    favorite_genres
                ),
                movie_catalog=movie_catalog,
                liked_movie_pseudo_rating=(
                    liked_movie_pseudo_rating
                ),
                favorite_genre_pseudo_rating=(
                    favorite_genre_pseudo_rating
                ),
            )
        )

        return self.recommend_with_profile(
            profile=profile,
            candidates=candidates,
            top_k=top_k,
            snapshot_year=snapshot_year,
            min_score=min_score,
        )

    def recommend_with_profile(
        self,
        *,
        profile: UserPreferenceProfile,
        candidates: Iterable[MovieRecord],
        top_k: Optional[int] = None,
        snapshot_year: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RecommendationResult:
        """
        Recommendation jika caller sudah mempunyai UserPreferenceProfile.

        Ini penting nanti supaya FastAPI/Laravel dapat memilih:
        - build profile pada request yang sama, atau
        - memakai profile yang sebelumnya sudah disimpan/cache.
        """
        candidate_list = list(
            candidates
        )

        if not candidate_list:
            return RecommendationResult(
                recommendations=[],
                candidate_count=0,
                returned_count=0,
                model_name=(
                    self.predictor.model_name
                ),
                profile_source=(
                    getattr(
                        profile,
                        "source",
                        "unknown",
                    )
                ),
                warnings=(
                    getattr(
                        profile,
                        "warnings",
                        None,
                    )
                ),
            )

        feature_batch = (
            self.feature_builder
            .build_matrix(
                profile=profile,
                candidates=candidate_list,
                snapshot_year=snapshot_year,
            )
        )

        prediction = (
            self.predictor
            .predict_feature_batch(
                feature_batch
            )
        )

        # Extra defensive contract:
        # predictor movie IDs harus sama dengan FeatureBuilder movie IDs.
        if (
            prediction.movie_ids
            != feature_batch.movie_ids
        ):
            raise RuntimeError(
                "Predictor mengembalikan movie_ids yang tidak sama "
                "dengan FeatureBatch."
            )

        recommendations = (
            self.ranker
            .rank_from_feature_batch(
                candidates=candidate_list,
                scores=prediction.scores,
                feature_batch=feature_batch,
                top_k=top_k,
                min_score=min_score,
            )
        )

        return RecommendationResult(
            recommendations=recommendations,
            candidate_count=len(
                candidate_list
            ),
            returned_count=len(
                recommendations
            ),
            model_name=(
                prediction.model_name
            ),
            profile_source=(
                getattr(
                    profile,
                    "source",
                    "unknown",
                )
            ),
            warnings=(
                getattr(
                    profile,
                    "warnings",
                    None,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Info / health-check helper
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """
        Informasi aman untuk debug/health-check API.
        """
        predictor_info = (
            self.predictor
            .model_info()
        )

        return {
            "service": (
                "aoranema_movie_recommendation"
            ),
            "model": predictor_info,
            "default_top_k": (
                self.ranker
                .default_top_k
            ),
            "pipeline": [
                "preference_builder",
                "feature_builder",
                "predictor",
                "ranker",
            ],
        }

    # ------------------------------------------------------------------
    # Internal contract validation
    # ------------------------------------------------------------------

    def _validate_component_contract(
        self,
    ) -> None:
        """
        Fail-fast jika file/config production tidak sinkron.
        """
        predictor_features = list(
            self.predictor
            .feature_names
        )

        builder_features = list(
            self.feature_builder
            .final_feature_names
        )

        if (
            predictor_features
            != builder_features
        ):
            raise ValueError(
                "FeatureBuilder dan Predictor memakai urutan feature berbeda."
            )

        if (
            self.predictor
            .expected_feature_count
            != len(
                builder_features
            )
        ):
            raise ValueError(
                "Jumlah feature FeatureBuilder dan Predictor tidak sama."
            )

        if (
            self.ranker
            .default_top_k
            != int(
                self.predictor
                .model_config[
                    "ranking_k"
                ]
            )
        ):
            raise ValueError(
                "Default Top-K Ranker tidak sama dengan model_config."
            )


def load_recommendation_service(
    ml_root: Optional[
        Union[str, Path]
    ] = None,
) -> RecommendationService:
    return RecommendationService.from_repo(
        ml_root=ml_root
    )


__all__ = [
    "RecommendationResult",
    "RecommendationService",
    "load_recommendation_service",
]
