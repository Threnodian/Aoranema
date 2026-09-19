"""
Aoranema - Recommendation Predictor
===================================

Memuat model XGBoost final dan mengubah feature matrix menjadi score ranking.

Alur:

    PreferenceBuilder
        ↓
    FeatureBuilder
        ↓
    RecommendationPredictor
        ↓
    raw ranking scores
        ↓
    MovieRanker
        ↓
    Top-N recommendation

PENTING:
- score XGBRanker BUKAN probabilitas,
- score boleh negatif,
- yang penting adalah urutannya: score lebih besar = ranking lebih tinggi,
- predictor selalu memakai urutan feature dari `feature_config.json`,
- predictor menghormati `best_iteration` model agar inference konsisten
  dengan XGBRanker saat training memakai early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union
import json

import numpy as np
import xgboost as xgb


MatrixLike = Union[
    np.ndarray,
    Sequence[Sequence[float]],
    Sequence[float],
]


def _load_json(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class PredictionBatch:
    """
    Hasil prediction untuk satu batch candidate.

    scores:
        Raw score XGBRanker. Bukan probabilitas.
    movie_ids:
        Opsional; diisi jika prediction berasal dari FeatureBatch.
    model_name:
        Nama model final dari model_config.
    """
    scores: np.ndarray
    movie_ids: Optional[List[Any]]
    model_name: str

    def __post_init__(self) -> None:
        if self.scores.ndim != 1:
            raise ValueError("Prediction scores harus 1D.")

        if not np.isfinite(self.scores).all():
            raise ValueError("Prediction scores mengandung NaN/Inf.")

        if (
            self.movie_ids is not None
            and len(self.movie_ids) != len(self.scores)
        ):
            raise ValueError(
                "Jumlah movie_ids tidak sama dengan jumlah scores."
            )

    @property
    def size(self) -> int:
        return int(self.scores.size)


class RecommendationPredictor:
    """
    Inference wrapper untuk model final Aoranema.

    Default repository:

        ml/
        ├── models/
        │   └── final_recommender_xgboost_ranker.json
        ├── config/
        │   ├── feature_config.json
        │   └── model_config.json
        └── src/
            └── predictor.py

    Contoh:

        predictor = RecommendationPredictor.from_repo()

        scores = predictor.predict(feature_batch.matrix)

    atau:

        prediction = predictor.predict_feature_batch(feature_batch)
        scores = prediction.scores
    """

    EXPECTED_FINAL_MODEL = "leakage_safe_full_content"

    def __init__(
        self,
        *,
        model_path: Union[str, Path],
        feature_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_config = dict(feature_config)
        self.model_config = dict(model_config)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file tidak ditemukan: {self.model_path}"
            )

        self.feature_names = list(
            self.feature_config["final_feature_names"]
        )

        self.expected_feature_count = len(
            self.feature_names
        )

        self.model_name = str(
            self.model_config.get(
                "selected_model",
                "",
            )
        )

        self._validate_config_contract()

        self.booster = xgb.Booster()
        self.booster.load_model(
            str(self.model_path)
        )

        self.model_feature_count = int(
            self.booster.num_features()
        )

        self.total_boosted_rounds = int(
            self.booster.num_boosted_rounds()
        )

        self.best_iteration = (
            self._read_best_iteration()
        )

        self.best_score = (
            self._read_best_score()
        )

        self._validate_model_contract()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        *,
        model_path: Union[str, Path],
        feature_config_path: Union[str, Path],
        model_config_path: Union[str, Path],
    ) -> "RecommendationPredictor":
        return cls(
            model_path=model_path,
            feature_config=_load_json(
                feature_config_path
            ),
            model_config=_load_json(
                model_config_path
            ),
        )

    @classmethod
    def from_repo(
        cls,
        ml_root: Optional[Union[str, Path]] = None,
    ) -> "RecommendationPredictor":
        if ml_root is None:
            ml_root = (
                Path(__file__)
                .resolve()
                .parents[1]
            )
        else:
            ml_root = Path(
                ml_root
            )

        return cls.from_files(
            model_path=(
                ml_root
                / "models"
                / "final_recommender_xgboost_ranker.json"
            ),
            feature_config_path=(
                ml_root
                / "config"
                / "feature_config.json"
            ),
            model_config_path=(
                ml_root
                / "config"
                / "model_config.json"
            ),
        )

    # Backward-friendly alias.
    from_repo_config = from_repo

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        features: MatrixLike,
    ) -> np.ndarray:
        """
        Predict raw ranking scores.

        Input:
            shape (n_candidates, n_features)

        Shortcut:
            1D feature vector sepanjang n_features dianggap satu candidate.

        Output:
            np.ndarray float32 shape (n_candidates,)
        """
        matrix = self._normalize_matrix(
            features
        )

        if matrix.shape[0] == 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        dmatrix = xgb.DMatrix(
            matrix
        )

        # XGBRanker sklearn wrapper menggunakan best_iteration saat model
        # dilatih dengan early stopping. Booster.predict() secara default
        # dapat memakai semua tree, jadi iteration_range diberikan eksplisit.
        kwargs = {}

        if self.best_iteration is not None:
            kwargs["iteration_range"] = (
                0,
                self.best_iteration + 1,
            )

        try:
            scores = self.booster.predict(
                dmatrix,
                **kwargs,
            )
        except TypeError:
            # Fallback untuk XGBoost lama yang belum mendukung
            # iteration_range pada Booster.predict.
            if self.best_iteration is not None:
                scores = self.booster.predict(
                    dmatrix,
                    ntree_limit=(
                        self.best_iteration + 1
                    ),
                )
            else:
                scores = self.booster.predict(
                    dmatrix
                )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != matrix.shape[0]:
            raise RuntimeError(
                "Jumlah output model berbeda dengan jumlah input row: "
                f"scores={len(scores)}, rows={matrix.shape[0]}."
            )

        if not np.isfinite(
            scores
        ).all():
            bad = np.flatnonzero(
                ~np.isfinite(scores)
            ).tolist()

            raise RuntimeError(
                "Model menghasilkan NaN/Inf pada index: "
                + ", ".join(
                    str(i)
                    for i in bad
                )
            )

        return scores

    def predict_one(
        self,
        features: Sequence[float],
    ) -> float:
        """
        Predict satu candidate dan return satu float score.
        """
        scores = self.predict(
            features
        )

        if len(scores) != 1:
            raise RuntimeError(
                "predict_one harus menghasilkan tepat satu score."
            )

        return float(
            scores[0]
        )

    def predict_feature_batch(
        self,
        feature_batch: Any,
    ) -> PredictionBatch:
        """
        Predict langsung dari output FeatureBuilder.

        Cross-check yang dilakukan:
        - feature_batch punya matrix,
        - feature_names sama PERSIS dan urutannya sama dengan config,
        - jumlah movie_ids sama dengan jumlah row.
        """
        if not hasattr(
            feature_batch,
            "matrix",
        ):
            raise TypeError(
                "feature_batch harus memiliki attribute 'matrix'."
            )

        if not hasattr(
            feature_batch,
            "feature_names",
        ):
            raise TypeError(
                "feature_batch harus memiliki attribute 'feature_names'."
            )

        if not hasattr(
            feature_batch,
            "movie_ids",
        ):
            raise TypeError(
                "feature_batch harus memiliki attribute 'movie_ids'."
            )

        batch_feature_names = list(
            feature_batch.feature_names
        )

        if batch_feature_names != self.feature_names:
            raise ValueError(
                "Urutan/nama feature pada FeatureBatch tidak sama "
                "dengan final_feature_names di feature_config.json."
            )

        matrix = self._normalize_matrix(
            feature_batch.matrix
        )

        movie_ids = list(
            feature_batch.movie_ids
        )

        if len(movie_ids) != matrix.shape[0]:
            raise ValueError(
                "Jumlah FeatureBatch.movie_ids tidak sama dengan "
                "jumlah row feature matrix."
            )

        scores = self.predict(
            matrix
        )

        return PredictionBatch(
            scores=scores,
            movie_ids=movie_ids,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_feature_batch(
        self,
        feature_batch: Any,
    ) -> None:
        """
        Hanya validasi tanpa menjalankan inference.
        """
        if not hasattr(
            feature_batch,
            "matrix",
        ):
            raise TypeError(
                "feature_batch harus memiliki attribute 'matrix'."
            )

        if not hasattr(
            feature_batch,
            "feature_names",
        ):
            raise TypeError(
                "feature_batch harus memiliki attribute 'feature_names'."
            )

        if list(
            feature_batch.feature_names
        ) != self.feature_names:
            raise ValueError(
                "FeatureBatch.feature_names tidak cocok dengan model."
            )

        matrix = self._normalize_matrix(
            feature_batch.matrix
        )

        if hasattr(
            feature_batch,
            "movie_ids",
        ):
            if len(
                list(
                    feature_batch.movie_ids
                )
            ) != matrix.shape[0]:
                raise ValueError(
                    "Jumlah movie_ids tidak sama dengan jumlah row."
                )

    def model_info(
        self,
    ) -> dict:
        """
        Metadata ringkas untuk health-check/debug API.
        Tidak berisi isi model.
        """
        return {
            "model_name": self.model_name,
            "model_file": self.model_path.name,
            "feature_count": self.expected_feature_count,
            "model_feature_count": self.model_feature_count,
            "total_boosted_rounds": self.total_boosted_rounds,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "scores_are_probabilities": False,
            "ranking_direction": "higher_score_is_better",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_matrix(
        self,
        features: MatrixLike,
    ) -> np.ndarray:
        try:
            matrix = np.asarray(
                features,
                dtype=np.float32,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Feature input harus dapat dikonversi menjadi numeric float32."
            ) from exc

        # Convenience: 149-vector -> one row.
        if matrix.ndim == 1:
            if matrix.size != self.expected_feature_count:
                raise ValueError(
                    "Feature vector 1D memiliki jumlah feature salah: "
                    f"expected={self.expected_feature_count}, "
                    f"got={matrix.size}."
                )

            matrix = matrix.reshape(
                1,
                -1,
            )

        if matrix.ndim != 2:
            raise ValueError(
                f"Feature matrix harus 2D, got shape={matrix.shape}."
            )

        if matrix.shape[1] != self.expected_feature_count:
            raise ValueError(
                "Jumlah feature tidak cocok dengan model: "
                f"expected={self.expected_feature_count}, "
                f"got={matrix.shape[1]}."
            )

        if not np.isfinite(
            matrix
        ).all():
            bad_rows, bad_cols = np.where(
                ~np.isfinite(
                    matrix
                )
            )

            preview = [
                (
                    int(r),
                    self.feature_names[
                        int(c)
                    ],
                )
                for r, c in zip(
                    bad_rows[:10],
                    bad_cols[:10],
                )
            ]

            raise ValueError(
                "Feature matrix mengandung NaN/Inf. "
                f"Contoh lokasi: {preview}"
            )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    def _validate_config_contract(
        self,
    ) -> None:
        if not self.feature_names:
            raise ValueError(
                "feature_config.final_feature_names kosong."
            )

        if len(
            set(
                self.feature_names
            )
        ) != len(
            self.feature_names
        ):
            raise ValueError(
                "feature_config.final_feature_names mengandung duplikat."
            )

        forbidden = {
            "movie_historical_rating_norm",
            "movie_historical_log_count_norm",
        }

        present = sorted(
            forbidden.intersection(
                self.feature_names
            )
        )

        if present:
            raise ValueError(
                "Final feature config masih mengandung "
                "temporal-risk feature: "
                + ", ".join(
                    present
                )
            )

        if (
            self.model_name
            != self.EXPECTED_FINAL_MODEL
        ):
            raise ValueError(
                "predictor.py ini dikunci untuk final model "
                f"'{self.EXPECTED_FINAL_MODEL}', tetapi model_config "
                f"memilih '{self.model_name}'."
            )

        ranking_k = self.model_config.get(
            "ranking_k"
        )

        if ranking_k is None:
            raise ValueError(
                "model_config kehilangan 'ranking_k'."
            )

        try:
            ranking_k = int(
                ranking_k
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "model_config['ranking_k'] harus integer."
            ) from exc

        if ranking_k <= 0:
            raise ValueError(
                "model_config['ranking_k'] harus > 0."
            )

    def _validate_model_contract(
        self,
    ) -> None:
        if (
            self.model_feature_count
            != self.expected_feature_count
        ):
            raise ValueError(
                "Jumlah feature model dan feature_config tidak sama: "
                f"model={self.model_feature_count}, "
                f"config={self.expected_feature_count}."
            )

        if self.total_boosted_rounds <= 0:
            raise ValueError(
                "Model XGBoost tidak memiliki boosted trees."
            )

        if self.best_iteration is not None:
            if self.best_iteration < 0:
                raise ValueError(
                    "best_iteration model tidak valid."
                )

            if (
                self.best_iteration
                >= self.total_boosted_rounds
            ):
                raise ValueError(
                    "best_iteration berada di luar jumlah boosted rounds: "
                    f"best_iteration={self.best_iteration}, "
                    f"rounds={self.total_boosted_rounds}."
                )

    def _read_best_iteration(
        self,
    ) -> Optional[int]:
        raw = self.booster.attr(
            "best_iteration"
        )

        if raw is None:
            return None

        try:
            return int(
                raw
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"best_iteration model tidak valid: {raw!r}"
            ) from exc

    def _read_best_score(
        self,
    ) -> Optional[float]:
        raw = self.booster.attr(
            "best_score"
        )

        if raw is None:
            return None

        try:
            return float(
                raw
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"best_score model tidak valid: {raw!r}"
            ) from exc


def load_predictor(
    ml_root: Optional[
        Union[str, Path]
    ] = None,
) -> RecommendationPredictor:
    return RecommendationPredictor.from_repo(
        ml_root=ml_root
    )


__all__ = [
    "PredictionBatch",
    "RecommendationPredictor",
    "load_predictor",
]
