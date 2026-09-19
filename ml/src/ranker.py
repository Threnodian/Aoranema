"""
Aoranema - Movie Ranker
=======================

Mengurutkan candidate film berdasarkan score dari model recommendation.

Alur production yang diharapkan:

    PreferenceBuilder
        ↓
    FeatureBuilder
        ↓
    Predictor
        ↓
    scores
        ↓
    MovieRanker
        ↓
    Top-N recommendation

File ini TIDAK:
- membuat preference user,
- membuat 149 feature,
- menjalankan model XGBoost.

Tugasnya hanya:
1. memvalidasi score,
2. mengurutkan dari score tertinggi ke terendah,
3. mengambil Top-K,
4. menambahkan `rank` dan `recommendation_score`,
5. tidak mengubah object candidate asli.

Tie handling:
Jika dua film mempunyai score sama, urutan input candidate dipertahankan
(stable ranking). Ini membuat hasil deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import json

import numpy as np


MovieRecord = Mapping[str, Any]


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class MovieRanker:
    """
    Ranker production untuk candidate film Aoranema.

    Contoh:

        ranker = MovieRanker.from_repo_config()

        ranked = ranker.rank(
            candidates=current_movies,
            scores=scores_from_predictor,
        )

    Output:
        [
            {
                ...candidate fields...,
                "rank": 1,
                "recommendation_score": 2.345
            },
            ...
        ]
    """

    MOVIE_ID_ALIASES = (
        "movie_id",
        "movieId",
        "id",
        "tmdbId",
        "tmdb_id",
    )

    def __init__(
        self,
        *,
        default_top_k: int = 10,
        rank_field: str = "rank",
        score_field: str = "recommendation_score",
        require_movie_id: bool = True,
    ) -> None:
        if not isinstance(default_top_k, int):
            raise TypeError("default_top_k harus integer.")

        if default_top_k <= 0:
            raise ValueError("default_top_k harus > 0.")

        if not rank_field:
            raise ValueError("rank_field tidak boleh kosong.")

        if not score_field:
            raise ValueError("score_field tidak boleh kosong.")

        self.default_top_k = default_top_k
        self.rank_field = str(rank_field)
        self.score_field = str(score_field)
        self.require_movie_id = bool(require_movie_id)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config_path: Union[str, Path],
        *,
        require_movie_id: bool = True,
    ) -> "MovieRanker":
        config = _load_json(model_config_path)

        ranking_k = config.get("ranking_k", 10)

        try:
            ranking_k = int(ranking_k)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model_config['ranking_k'] harus integer."
            ) from exc

        return cls(
            default_top_k=ranking_k,
            require_movie_id=require_movie_id,
        )

    @classmethod
    def from_repo_config(
        cls,
        ml_root: Optional[Union[str, Path]] = None,
        *,
        require_movie_id: bool = True,
    ) -> "MovieRanker":
        """
        Default repository:

            ml/
            ├── config/
            │   └── model_config.json
            └── src/
                └── ranker.py
        """
        if ml_root is None:
            ml_root = Path(__file__).resolve().parents[1]
        else:
            ml_root = Path(ml_root)

        return cls.from_model_config(
            ml_root / "config" / "model_config.json",
            require_movie_id=require_movie_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: Iterable[MovieRecord],
        scores: Union[
            Sequence[float],
            np.ndarray,
        ],
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Mengurutkan candidate berdasarkan score descending.

        Parameters
        ----------
        candidates:
            Candidate movie records dalam urutan yang sama dengan rows
            yang dikirim ke Predictor.

        scores:
            Satu score per candidate.
            Menerima shape:
                (n,)
                (n, 1)

        top_k:
            Berapa film teratas yang dikembalikan.
            Jika None, gunakan ranking_k dari model_config (default 10).

        min_score:
            Opsional. Candidate dengan score di bawah nilai ini dibuang
            sebelum Top-K diambil.

            Secara default None karena score XGBRanker bukan probabilitas
            dan tidak mempunyai threshold universal seperti 0.5.

        Returns
        -------
        List[dict]
            Copy dari candidate record +:
                rank
                recommendation_score
        """
        candidate_list = list(candidates)

        if not candidate_list:
            score_array = self._normalize_scores(
                scores,
                expected_count=0,
            )

            if score_array.size != 0:
                raise ValueError(
                    "Candidates kosong tetapi scores tidak kosong."
                )

            return []

        for index, candidate in enumerate(candidate_list):
            if not isinstance(candidate, Mapping):
                raise TypeError(
                    f"Candidate index={index} harus Mapping/dict."
                )

            if self.require_movie_id:
                self._extract_movie_id(
                    candidate,
                    index=index,
                )

        score_array = self._normalize_scores(
            scores,
            expected_count=len(candidate_list),
        )

        resolved_top_k = self._resolve_top_k(
            top_k
        )

        if min_score is not None:
            try:
                min_score = float(min_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "min_score harus numeric atau None."
                ) from exc

            if not np.isfinite(min_score):
                raise ValueError(
                    "min_score harus finite."
                )

        # Stable descending sort:
        # score sama -> order input tetap sama.
        order = np.argsort(
            -score_array,
            kind="stable",
        )

        ranked: List[Dict[str, Any]] = []

        for source_index in order:
            source_index = int(source_index)
            score = float(
                score_array[source_index]
            )

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            candidate_copy = dict(
                candidate_list[source_index]
            )

            candidate_copy[
                self.rank_field
            ] = len(ranked) + 1

            candidate_copy[
                self.score_field
            ] = score

            ranked.append(
                candidate_copy
            )

            if len(ranked) >= resolved_top_k:
                break

        return ranked

    def rank_compact(
        self,
        movie_ids: Sequence[Any],
        scores: Union[
            Sequence[float],
            np.ndarray,
        ],
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Versi compact jika caller hanya membutuhkan ID, rank, dan score.

        Output:
            [
                {
                    "movie_id": ...,
                    "rank": 1,
                    "recommendation_score": ...
                }
            ]
        """
        candidates = [
            {"movie_id": movie_id}
            for movie_id in movie_ids
        ]

        return self.rank(
            candidates=candidates,
            scores=scores,
            top_k=top_k,
            min_score=min_score,
        )

    def rank_from_feature_batch(
        self,
        *,
        candidates: Iterable[MovieRecord],
        scores: Union[
            Sequence[float],
            np.ndarray,
        ],
        feature_batch: Any,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extra safety check saat dipakai bersama FeatureBuilder.

        Memastikan:
        - jumlah candidate = jumlah row feature,
        - urutan movie ID candidate sama dengan feature_batch.movie_ids.

        `feature_batch` sengaja duck-typed agar ranker.py tidak harus
        meng-import feature_builder.py dan tidak membuat coupling/circular import.
        """
        candidate_list = list(candidates)

        if not hasattr(feature_batch, "movie_ids"):
            raise TypeError(
                "feature_batch harus memiliki attribute 'movie_ids'."
            )

        if not hasattr(feature_batch, "matrix"):
            raise TypeError(
                "feature_batch harus memiliki attribute 'matrix'."
            )

        feature_movie_ids = list(
            feature_batch.movie_ids
        )

        matrix = np.asarray(
            feature_batch.matrix
        )

        if matrix.ndim != 2:
            raise ValueError(
                "feature_batch.matrix harus 2D."
            )

        if matrix.shape[0] != len(
            candidate_list
        ):
            raise ValueError(
                "Jumlah candidate berbeda dengan jumlah row feature_batch."
            )

        if len(feature_movie_ids) != len(
            candidate_list
        ):
            raise ValueError(
                "feature_batch.movie_ids berbeda jumlah dengan candidate."
            )

        candidate_movie_ids = [
            self._extract_movie_id(
                movie,
                index=index,
            )
            for index, movie
            in enumerate(candidate_list)
        ]

        if candidate_movie_ids != feature_movie_ids:
            raise ValueError(
                "Urutan/ID candidate tidak sama dengan feature_batch.movie_ids. "
                "Jangan ranking score terhadap candidate yang berbeda urutan."
            )

        return self.rank(
            candidates=candidate_list,
            scores=scores,
            top_k=top_k,
            min_score=min_score,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _normalize_scores(
        self,
        scores: Union[
            Sequence[float],
            np.ndarray,
        ],
        *,
        expected_count: int,
    ) -> np.ndarray:
        try:
            array = np.asarray(
                scores,
                dtype=np.float32,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "scores harus dapat dikonversi menjadi angka."
            ) from exc

        # XGBoost normal -> (n,)
        # Beberapa wrapper bisa -> (n,1)
        if array.ndim == 2:
            if array.shape[1] != 1:
                raise ValueError(
                    "scores 2D hanya didukung jika shape=(n, 1)."
                )
            array = array[:, 0]

        if array.ndim != 1:
            raise ValueError(
                f"scores harus 1D atau (n,1), got shape={array.shape}."
            )

        if len(array) != expected_count:
            raise ValueError(
                "Jumlah score tidak sama dengan jumlah candidate: "
                f"scores={len(array)}, candidates={expected_count}."
            )

        if not np.isfinite(array).all():
            bad = np.flatnonzero(
                ~np.isfinite(array)
            ).tolist()

            raise ValueError(
                "scores mengandung NaN/Inf pada index: "
                + ", ".join(
                    str(i)
                    for i in bad
                )
            )

        return array

    def _resolve_top_k(
        self,
        top_k: Optional[int],
    ) -> int:
        if top_k is None:
            return self.default_top_k

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k harus integer atau None."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k harus > 0."
            )

        return top_k

    def _extract_movie_id(
        self,
        candidate: MovieRecord,
        *,
        index: int,
    ) -> Any:
        for key in self.MOVIE_ID_ALIASES:
            if key in candidate:
                value = candidate[key]

                if value is not None:
                    if (
                        isinstance(value, str)
                        and not value.strip()
                    ):
                        continue

                    return value

        raise ValueError(
            f"Candidate index={index} tidak memiliki movie ID. "
            f"Gunakan salah satu field: {self.MOVIE_ID_ALIASES}"
        )


def load_ranker(
    ml_root: Optional[
        Union[str, Path]
    ] = None,
) -> MovieRanker:
    return MovieRanker.from_repo_config(
        ml_root=ml_root
    )


__all__ = [
    "MovieRanker",
    "load_ranker",
]
