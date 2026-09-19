"""
Aoranema - User Preference Builder
===================================

Membangun profil preferensi user yang kompatibel dengan model final
Personalized Movie Recommendation V2.

File ini TIDAK melakukan prediksi/ranking.
Tugasnya hanya menjawab:

    "User ini sebenarnya suka apa?"

Output dari modul ini nantinya dipakai oleh feature_builder.py.

Training-faithful logic:
- genre preference memakai smoothed mean rating terhadap mean rating user,
- genre relative preference membandingkan genre dengan baseline rating user,
- genre evidence menyimpan banyaknya bukti per genre,
- metadata preference dibentuk dari histori rating yang sudah di-center
  terhadap mean rating user,
- metadata IDs di-hash memakai CRC32 yang sama dengan notebook final.

Catatan cold-start:
- onboarding liked movies dapat diubah menjadi pseudo-rating,
- onboarding favorite genres dapat diberi pseudo evidence,
- ini adalah strategi deployment untuk user baru; bukan rating MovieLens asli.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import json
import math
import zlib

import numpy as np


Number = Union[int, float]
MovieRecord = Mapping[str, Any]


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Interaction:
    """
    Satu histori explicit preference user.

    Parameters
    ----------
    movie_id:
        ID film pada katalog aplikasi/metadata yang dipakai builder.
    rating:
        Skala 0.5 - 5.0 agar konsisten dengan training MovieLens.
    """
    movie_id: Union[int, str]
    rating: float


@dataclass
class UserPreferenceProfile:
    """
    Profil user yang siap diteruskan ke feature_builder.py.
    """

    history_count: int
    user_mean_rating: float

    # Per-genre.
    genre_abs_pref: Dict[str, float]
    genre_rel_pref: Dict[str, float]
    genre_evidence_count: Dict[str, float]
    genre_log_count: Dict[str, float]
    genre_known: Dict[str, float]

    # Raw hashed preference vector per metadata field.
    # Vector ini belum dinormalisasi; feature_builder nantinya menghitung
    # cosine similarity terhadap candidate movie vector.
    metadata_profiles: Dict[str, List[float]]

    # Membantu debugging / cold-start handling.
    source: str = "history"
    warnings: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(
        self,
        path: Union[str, Path],
        *,
        indent: int = 2,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=indent,
                ensure_ascii=False,
            )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(value: Any) -> List[Any]:
    """
    Mengubah list/tuple/JSON-string/scalar menjadi list Python.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        # JSON list: "[1, 2, 3]"
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # MovieLens genres: "Action|Sci-Fi"
        if "|" in text:
            return [
                item.strip()
                for item in text.split("|")
                if item.strip()
            ]

        return [text]

    return [value]


def _normalize_rating_01(
    rating: Number,
    rating_min: float,
    rating_max: float,
) -> float:
    rating_range = rating_max - rating_min
    if rating_range <= 0:
        raise ValueError("rating_max harus lebih besar dari rating_min.")

    value = (float(rating) - rating_min) / rating_range
    return float(np.clip(value, 0.0, 1.0))


def _stable_bucket(
    value: Any,
    salt: str,
    dim: int,
) -> int:
    """
    Harus identik dengan notebook:
        crc32(f"{salt}:{entity_id}") % dim
    """
    payload = f"{salt}:{value}".encode("utf-8")
    return zlib.crc32(payload) % dim


def _l2_normalized_hashed_vector(
    entity_ids: Sequence[Any],
    *,
    dim: int,
    salt: str,
) -> np.ndarray:
    """
    Membuat hashed vector satu film lalu L2 normalize,
    sama seperti movie hashed matrix pada notebook final.
    """
    vector = np.zeros(dim, dtype=np.float32)

    for entity_id in entity_ids:
        if entity_id is None:
            continue

        # Hindari NaN float.
        if isinstance(entity_id, float) and math.isnan(entity_id):
            continue

        bucket = _stable_bucket(
            entity_id,
            salt,
            dim,
        )
        vector[bucket] += 1.0

    norm = float(np.linalg.norm(vector))

    if norm > 0.0:
        vector /= norm

    return vector


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

class PreferenceBuilder:
    """
    Builder profil preference user untuk Aoranema.

    Recommended usage di repo:

        builder = PreferenceBuilder.from_repo_config()

        profile = builder.build_profile(
            interactions=[
                {"movie_id": 1, "rating": 5.0},
                {"movie_id": 2, "rating": 4.0},
            ],
            movie_catalog=movie_catalog,
        )

    movie_catalog dapat berupa:
        {
            1: {
                "movieId": 1,
                "genres": "Adventure|Animation|Children",
                "director_ids_json": "[7879]",
                ...
            },
            ...
        }

    atau iterable/list record film.
    """

    # Alias field metadata agar input production lebih fleksibel.
    ENTITY_FIELD_ALIASES = {
        "director": (
            "director_ids_json",
            "director_ids",
            "directors",
        ),
        "writer": (
            "writer_ids_json",
            "writer_ids",
            "writers",
        ),
        "cast": (
            "top_cast_ids_json",
            "top_cast_ids",
            "cast_ids",
            "cast",
        ),
        "keyword": (
            "keyword_ids_json",
            "keyword_ids",
            "keywords",
        ),
        "company": (
            "production_company_ids_json",
            "production_company_ids",
            "company_ids",
            "companies",
        ),
        "collection": (
            "collection_ids_json",
            "collection_ids",
            "collection_id",
        ),
    }

    GENRE_FIELD_ALIASES = (
        "genres",
        "movielens_genres",
        "genre_names",
    )

    MOVIE_ID_ALIASES = (
        "movie_id",
        "movieId",
        "id",
    )

    def __init__(
        self,
        *,
        genres: Sequence[str],
        hash_config: Mapping[str, Mapping[str, Any]],
        genre_smoothing_alpha: float = 5.0,
        rating_min: float = 0.5,
        rating_max: float = 5.0,
        default_user_mean_rating: float = 3.5,
    ) -> None:
        self.genres = list(genres)
        self.genre_set = set(self.genres)
        self.hash_config = {
            field: dict(cfg)
            for field, cfg in hash_config.items()
        }

        self.genre_smoothing_alpha = float(
            genre_smoothing_alpha
        )

        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.rating_range = (
            self.rating_max - self.rating_min
        )

        if self.rating_range <= 0:
            raise ValueError(
                "rating_max harus lebih besar dari rating_min."
            )

        self.default_user_mean_rating = float(
            default_user_mean_rating
        )

        if not (
            self.rating_min
            <= self.default_user_mean_rating
            <= self.rating_max
        ):
            raise ValueError(
                "default_user_mean_rating harus berada "
                "di dalam skala rating."
            )

    # -----------------------------------------------------------------
    # Construction from repository configs
    # -----------------------------------------------------------------

    @classmethod
    def from_config_files(
        cls,
        feature_config_path: Union[str, Path],
        preference_config_path: Union[str, Path],
    ) -> "PreferenceBuilder":
        feature_config = _load_json(
            feature_config_path
        )
        preference_config = _load_json(
            preference_config_path
        )

        normalization = feature_config.get(
            "normalization",
            {},
        )

        return cls(
            genres=feature_config["genres"],
            hash_config=feature_config["hash_config"],
            genre_smoothing_alpha=preference_config.get(
                "genre_smoothing_alpha",
                5.0,
            ),
            rating_min=normalization.get(
                "rating_min",
                0.5,
            ),
            rating_max=normalization.get(
                "rating_max",
                5.0,
            ),
        )

    @classmethod
    def from_repo_config(
        cls,
        ml_root: Optional[Union[str, Path]] = None,
    ) -> "PreferenceBuilder":
        """
        Default layout:

            ml/
            ├── config/
            │   ├── feature_config.json
            │   └── preference_config.json
            └── src/
                └── preference_builder.py
        """
        if ml_root is None:
            ml_root = Path(__file__).resolve().parents[1]
        else:
            ml_root = Path(ml_root)

        return cls.from_config_files(
            ml_root / "config" / "feature_config.json",
            ml_root / "config" / "preference_config.json",
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def build_profile(
        self,
        interactions: Iterable[
            Union[Interaction, Mapping[str, Any]]
        ],
        movie_catalog: Union[
            Mapping[Union[int, str], MovieRecord],
            Iterable[MovieRecord],
        ],
        *,
        favorite_genres: Optional[Sequence[str]] = None,
        favorite_genre_pseudo_rating: float = 4.5,
        favorite_genre_evidence: float = 1.0,
        source: str = "history",
    ) -> UserPreferenceProfile:
        """
        Membangun profil preference.

        Parameters
        ----------
        interactions:
            Histori explicit rating:
                {"movie_id": 123, "rating": 4.5}

        movie_catalog:
            Metadata film keyed by movie ID atau list record.

        favorite_genres:
            Opsional untuk onboarding/cold-start.
            Genre pilihan user diberi pseudo evidence.
            Ini deployment heuristic, bukan rating MovieLens asli.

        favorite_genre_pseudo_rating:
            Nilai pseudo-rating untuk genre onboarding.

        favorite_genre_evidence:
            Berapa evidence pseudo untuk favorite genre.

        source:
            Label untuk debugging, misalnya:
            "history", "onboarding", atau "history+onboarding".
        """
        catalog = self._normalize_catalog(
            movie_catalog
        )

        clean_interactions = self._normalize_interactions(
            interactions
        )

        warnings: List[str] = []

        # --------------------------------------------------------------
        # 1. Mean rating user
        # --------------------------------------------------------------
        known_ratings: List[float] = []

        for item in clean_interactions:
            if item.movie_id not in catalog:
                warnings.append(
                    f"movie_id={item.movie_id} tidak ditemukan "
                    "di movie_catalog; interaction dilewati."
                )
                continue

            known_ratings.append(
                float(item.rating)
            )

        if known_ratings:
            user_mean = float(
                np.mean(
                    np.asarray(
                        known_ratings,
                        dtype=np.float32,
                    )
                )
            )
        else:
            user_mean = (
                self.default_user_mean_rating
            )

        # --------------------------------------------------------------
        # 2. Genre accumulators
        # --------------------------------------------------------------
        genre_sum = {
            genre: 0.0
            for genre in self.genres
        }

        genre_count = {
            genre: 0.0
            for genre in self.genres
        }

        # --------------------------------------------------------------
        # 3. Metadata raw preference vectors
        # --------------------------------------------------------------
        metadata_profiles = {
            field: np.zeros(
                int(cfg["dim"]),
                dtype=np.float32,
            )
            for field, cfg
            in self.hash_config.items()
        }

        valid_history_count = 0

        for item in clean_interactions:
            movie = catalog.get(
                item.movie_id
            )

            if movie is None:
                continue

            rating = float(
                item.rating
            )

            valid_history_count += 1

            # Genre evidence menggunakan raw rating,
            # identik dengan training profile.
            for genre in self._extract_genres(
                movie
            ):
                if genre not in self.genre_set:
                    continue

                genre_sum[genre] += rating
                genre_count[genre] += 1.0

            # Metadata history weight:
            # (rating - user_mean) / rating_range
            centered_weight = (
                rating - user_mean
            ) / self.rating_range

            for field, cfg in self.hash_config.items():
                entity_ids = (
                    self._extract_entity_ids(
                        movie,
                        field,
                    )
                )

                if not entity_ids:
                    continue

                movie_vector = (
                    _l2_normalized_hashed_vector(
                        entity_ids,
                        dim=int(cfg["dim"]),
                        salt=str(cfg["salt"]),
                    )
                )

                metadata_profiles[field] += (
                    float(centered_weight)
                    * movie_vector
                )

        # --------------------------------------------------------------
        # 4. Optional onboarding favorite genres
        # --------------------------------------------------------------
        if favorite_genres:
            pseudo_rating = float(
                np.clip(
                    favorite_genre_pseudo_rating,
                    self.rating_min,
                    self.rating_max,
                )
            )

            pseudo_evidence = max(
                0.0,
                float(
                    favorite_genre_evidence
                ),
            )

            for genre in favorite_genres:
                if genre not in self.genre_set:
                    warnings.append(
                        f"Genre onboarding '{genre}' tidak dikenal "
                        "oleh model dan diabaikan."
                    )
                    continue

                genre_sum[genre] += (
                    pseudo_rating
                    * pseudo_evidence
                )
                genre_count[genre] += (
                    pseudo_evidence
                )

        # --------------------------------------------------------------
        # 5. Smoothed genre preference
        # --------------------------------------------------------------
        genre_abs_pref: Dict[str, float] = {}
        genre_rel_pref: Dict[str, float] = {}
        genre_log_count: Dict[str, float] = {}
        genre_known: Dict[str, float] = {}

        alpha = (
            self.genre_smoothing_alpha
        )

        for genre in self.genres:
            count = float(
                genre_count[genre]
            )
            total = float(
                genre_sum[genre]
            )

            smoothed_mean = (
                total
                + alpha * user_mean
            ) / (
                count + alpha
            )

            genre_abs_pref[genre] = (
                _normalize_rating_01(
                    smoothed_mean,
                    self.rating_min,
                    self.rating_max,
                )
            )

            relative = (
                smoothed_mean
                - user_mean
            ) / self.rating_range

            genre_rel_pref[genre] = float(
                np.clip(
                    relative,
                    -1.0,
                    1.0,
                )
            )

            genre_log_count[genre] = (
                float(
                    np.log1p(
                        count
                    )
                )
            )

            genre_known[genre] = (
                1.0
                if count > 0.0
                else 0.0
            )

        # JSON serializable vectors.
        serialized_metadata_profiles = {
            field: vector.astype(
                np.float32
            ).tolist()
            for field, vector
            in metadata_profiles.items()
        }

        if (
            valid_history_count == 0
            and not favorite_genres
        ):
            warnings.append(
                "Tidak ada histori atau onboarding preference. "
                "Profile menggunakan neutral/default prior."
            )

        return UserPreferenceProfile(
            history_count=valid_history_count,
            user_mean_rating=user_mean,
            genre_abs_pref=genre_abs_pref,
            genre_rel_pref=genre_rel_pref,
            genre_evidence_count={
                genre: float(
                    genre_count[genre]
                )
                for genre in self.genres
            },
            genre_log_count=genre_log_count,
            genre_known=genre_known,
            metadata_profiles=(
                serialized_metadata_profiles
            ),
            source=source,
            warnings=warnings or None,
        )

    def build_onboarding_profile(
        self,
        *,
        favorite_movie_ids: Sequence[
            Union[int, str]
        ],
        favorite_genres: Sequence[str],
        movie_catalog: Union[
            Mapping[Union[int, str], MovieRecord],
            Iterable[MovieRecord],
        ],
        liked_movie_pseudo_rating: float = 4.5,
        favorite_genre_pseudo_rating: float = 4.5,
    ) -> UserPreferenceProfile:
        """
        Helper cold-start untuk user baru.

        Film yang dipilih user sebagai "suka" di onboarding diperlakukan
        sebagai pseudo-rating. Ini hanya strategi deployment untuk membuat
        profil awal sebelum user punya explicit rating di Aoranema.
        """
        interactions = [
            Interaction(
                movie_id=movie_id,
                rating=liked_movie_pseudo_rating,
            )
            for movie_id in favorite_movie_ids
        ]

        return self.build_profile(
            interactions=interactions,
            movie_catalog=movie_catalog,
            favorite_genres=favorite_genres,
            favorite_genre_pseudo_rating=(
                favorite_genre_pseudo_rating
            ),
            source="onboarding",
        )

    # -----------------------------------------------------------------
    # Input normalization
    # -----------------------------------------------------------------

    def _normalize_interactions(
        self,
        interactions: Iterable[
            Union[
                Interaction,
                Mapping[str, Any],
            ]
        ],
    ) -> List[Interaction]:
        result: List[Interaction] = []

        for raw in interactions:
            if isinstance(
                raw,
                Interaction,
            ):
                interaction = raw
            else:
                movie_id = (
                    raw.get("movie_id")
                    if "movie_id" in raw
                    else raw.get("movieId")
                )

                if movie_id is None:
                    raise ValueError(
                        "Interaction harus memiliki "
                        "'movie_id' atau 'movieId'."
                    )

                if "rating" not in raw:
                    raise ValueError(
                        "Interaction harus memiliki 'rating'."
                    )

                interaction = Interaction(
                    movie_id=movie_id,
                    rating=float(
                        raw["rating"]
                    ),
                )

            rating = float(
                interaction.rating
            )

            if not (
                self.rating_min
                <= rating
                <= self.rating_max
            ):
                raise ValueError(
                    f"Rating {rating} di luar skala "
                    f"{self.rating_min}-{self.rating_max}."
                )

            result.append(
                interaction
            )

        return result

    def _normalize_catalog(
        self,
        movie_catalog: Union[
            Mapping[
                Union[int, str],
                MovieRecord,
            ],
            Iterable[MovieRecord],
        ],
    ) -> Dict[Union[int, str], MovieRecord]:
        if isinstance(
            movie_catalog,
            Mapping,
        ):
            return dict(
                movie_catalog
            )

        result: Dict[
            Union[int, str],
            MovieRecord,
        ] = {}

        for movie in movie_catalog:
            movie_id = None

            for key in self.MOVIE_ID_ALIASES:
                if key in movie:
                    movie_id = movie[key]
                    break

            if movie_id is None:
                raise ValueError(
                    "Movie record harus memiliki salah satu "
                    f"ID field: {self.MOVIE_ID_ALIASES}"
                )

            result[movie_id] = movie

        return result

    def _extract_genres(
        self,
        movie: MovieRecord,
    ) -> List[str]:
        for field in self.GENRE_FIELD_ALIASES:
            if field not in movie:
                continue

            genres = _as_list(
                movie[field]
            )

            # Kalau JSON string names / list masuk,
            # hasil sudah berupa list.
            return [
                str(g).strip()
                for g in genres
                if str(g).strip()
                and str(g).strip()
                != "(no genres listed)"
            ]

        return []

    def _extract_entity_ids(
        self,
        movie: MovieRecord,
        field: str,
    ) -> List[Any]:
        aliases = (
            self.ENTITY_FIELD_ALIASES.get(
                field,
                (),
            )
        )

        for alias in aliases:
            if alias not in movie:
                continue

            value = movie[alias]

            # collection_id biasanya scalar.
            ids = _as_list(
                value
            )

            clean_ids: List[Any] = []

            for entity_id in ids:
                if entity_id is None:
                    continue

                if (
                    isinstance(
                        entity_id,
                        float,
                    )
                    and math.isnan(
                        entity_id
                    )
                ):
                    continue

                clean_ids.append(
                    entity_id
                )

            return clean_ids

        # Fallback ke column name dari feature_config.
        cfg = self.hash_config.get(
            field,
            {},
        )

        configured_column = cfg.get(
            "column"
        )

        if (
            configured_column
            and configured_column in movie
        ):
            return _as_list(
                movie[
                    configured_column
                ]
            )

        return []


# ---------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------

def load_preference_builder(
    ml_root: Optional[
        Union[str, Path]
    ] = None,
) -> PreferenceBuilder:
    """
    Shortcut:
        builder = load_preference_builder()
    """
    return PreferenceBuilder.from_repo_config(
        ml_root=ml_root
    )


__all__ = [
    "Interaction",
    "UserPreferenceProfile",
    "PreferenceBuilder",
    "load_preference_builder",
]
