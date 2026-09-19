"""
Aoranema - Feature Builder
==========================

Mengubah:
    UserPreferenceProfile + metadata film kandidat
menjadi feature matrix float32 dengan urutan kolom yang PERSIS sama dengan
`config/feature_config.json`.

File ini mengikuti kontrak model final:
    leakage_safe_full_content

Feature yang sengaja TIDAK dipakai model final:
    - movie_historical_rating_norm
    - movie_historical_log_count_norm

Tugas file ini hanya membangun input model. Training, prediksi, dan ranking
ditangani file lain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math
import re
import zlib

import numpy as np


ProfileLike = Union[Mapping[str, Any], Any]
MovieRecord = Mapping[str, Any]


@dataclass
class FeatureBatch:
    matrix: np.ndarray
    movie_ids: List[Any]
    feature_names: List[str]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _as_list(value: Any) -> List[Any]:
    if _is_missing(value):
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

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        if "|" in text:
            return [x.strip() for x in text.split("|") if x.strip()]

        return [text]

    return [value]


def _stable_bucket(value: Any, salt: str, dim: int) -> int:
    payload = f"{salt}:{value}".encode("utf-8")
    return zlib.crc32(payload) % dim


def _hashed_movie_vector(
    entity_ids: Sequence[Any],
    *,
    dim: int,
    salt: str,
) -> np.ndarray:
    """
    Sama dengan notebook:
    1. hash entity ID ke bucket,
    2. jumlahkan jika collision,
    3. L2 normalize per film.
    """
    vector = np.zeros(dim, dtype=np.float32)

    for entity_id in entity_ids:
        if _is_missing(entity_id):
            continue
        vector[_stable_bucket(entity_id, salt, dim)] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm

    return vector


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class FeatureBuilder:
    """
    Feature builder production untuk model recommendation final Aoranema.
    """

    MOVIE_ID_ALIASES = (
        "movie_id", "movieId", "id", "tmdbId", "tmdb_id"
    )

    GENRE_ALIASES = (
        "genres",
        "movielens_genres",
        "genre_names",
        "genre_names_json",
        "tmdb_genre_names_json",
        "tmdb_genres",
    )

    ENTITY_ALIASES = {
        "director": ("director_ids_json", "director_ids", "directors"),
        "writer": ("writer_ids_json", "writer_ids", "writers"),
        "cast": ("top_cast_ids_json", "top_cast_ids", "cast_ids", "cast"),
        "keyword": ("keyword_ids_json", "keyword_ids", "keywords"),
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

    COUNT_ALIASES = {
        "director": ("n_directors",),
        "writer": ("n_writers",),
        "cast": ("n_top_cast", "n_cast"),
        "keyword": ("n_keywords",),
    }

    # TMDB -> vocabulary MovieLens yang dipakai saat training.
    TMDB_GENRE_MAP = {
        "science fiction": "Sci-Fi",
        "family": "Children",
        "music": "Musical",
    }

    def __init__(
        self,
        *,
        feature_config: Mapping[str, Any],
        model_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.feature_config = dict(feature_config)
        self.model_config = dict(model_config) if model_config else None

        self.genres = list(self.feature_config["genres"])
        self.genre_slugs = list(
            self.feature_config.get(
                "genre_slugs",
                [_slug(g) for g in self.genres],
            )
        )

        if len(self.genres) != len(self.genre_slugs):
            raise ValueError("genres dan genre_slugs tidak sinkron.")

        self.genre_by_lower = {g.lower(): g for g in self.genres}
        self.hash_config = {
            k: dict(v)
            for k, v in self.feature_config["hash_config"].items()
        }
        self.affinity_fields = list(self.hash_config.keys())

        self.top_languages = list(self.feature_config["top_languages"])
        self.language_feature_names = list(
            self.feature_config["language_feature_names"]
        )
        self.final_feature_names = list(
            self.feature_config["final_feature_names"]
        )

        norm = self.feature_config["normalization"]
        self.rating_min = float(norm["rating_min"])
        self.rating_max = float(norm["rating_max"])
        self.rating_range = self.rating_max - self.rating_min

        self.year_center = float(norm["year_center"])
        self.year_scale = float(norm["year_scale"])
        self.movie_age_max = float(norm["movie_age_max"])

        self.runtime_center = float(norm["runtime_center"])
        self.runtime_scale = float(norm["runtime_scale"])

        self.genre_evidence_log_clip = float(
            norm["genre_evidence_log_clip"]
        )
        self.user_history_log_clip = float(
            norm["user_history_log_clip"]
        )

        self._validate_config_contract()

    @classmethod
    def from_config_files(
        cls,
        feature_config_path: Union[str, Path],
        model_config_path: Optional[Union[str, Path]] = None,
    ) -> "FeatureBuilder":
        feature_config = _load_json(feature_config_path)

        model_config = None
        if model_config_path is not None and Path(model_config_path).exists():
            model_config = _load_json(model_config_path)

        return cls(
            feature_config=feature_config,
            model_config=model_config,
        )

    @classmethod
    def from_repo_config(
        cls,
        ml_root: Optional[Union[str, Path]] = None,
    ) -> "FeatureBuilder":
        if ml_root is None:
            ml_root = Path(__file__).resolve().parents[1]
        else:
            ml_root = Path(ml_root)

        return cls.from_config_files(
            ml_root / "config" / "feature_config.json",
            ml_root / "config" / "model_config.json",
        )

    def build_matrix(
        self,
        profile: ProfileLike,
        candidates: Iterable[MovieRecord],
        *,
        snapshot_year: Optional[int] = None,
    ) -> FeatureBatch:
        profile_dict = self._normalize_profile(profile)

        if snapshot_year is None:
            snapshot_year = datetime.now(timezone.utc).year
        snapshot_year = int(snapshot_year)

        rows: List[np.ndarray] = []
        movie_ids: List[Any] = []

        for index, movie in enumerate(candidates):
            if not isinstance(movie, Mapping):
                raise TypeError("Setiap candidate harus berupa dict/Mapping.")

            feature_dict = self.build_feature_dict(
                profile_dict,
                movie,
                snapshot_year=snapshot_year,
            )

            row = np.asarray(
                [feature_dict[name] for name in self.final_feature_names],
                dtype=np.float32,
            )

            if not np.isfinite(row).all():
                bad = [
                    self.final_feature_names[int(i)]
                    for i in np.flatnonzero(~np.isfinite(row))
                ]
                raise ValueError(
                    f"Candidate index={index} menghasilkan NaN/Inf: {bad}"
                )

            rows.append(row)
            movie_ids.append(self._movie_id(movie, fallback=index))

        if rows:
            matrix = np.vstack(rows).astype(np.float32, copy=False)
        else:
            matrix = np.empty(
                (0, len(self.final_feature_names)),
                dtype=np.float32,
            )

        self.validate_matrix(matrix)

        return FeatureBatch(
            matrix=matrix,
            movie_ids=movie_ids,
            feature_names=list(self.final_feature_names),
        )

    def build_one(
        self,
        profile: ProfileLike,
        movie: MovieRecord,
        *,
        snapshot_year: Optional[int] = None,
    ) -> np.ndarray:
        return self.build_matrix(
            profile,
            [movie],
            snapshot_year=snapshot_year,
        ).matrix[0]

    def build_feature_dict(
        self,
        profile: ProfileLike,
        movie: MovieRecord,
        *,
        snapshot_year: Optional[int] = None,
    ) -> Dict[str, float]:
        p = self._normalize_profile(profile)

        if snapshot_year is None:
            snapshot_year = datetime.now(timezone.utc).year

        snapshot_year = int(snapshot_year)
        f: Dict[str, float] = {}

        genre_abs = p["genre_abs_pref"]
        genre_rel = p["genre_rel_pref"]
        genre_log = p["genre_log_count"]
        genre_known = p["genre_known"]

        movie_genres = set(self._genres(movie))

        # 1-3: abs preference, movie genre, interaction.
        for genre, slug in zip(self.genres, self.genre_slugs):
            pref = float(genre_abs[genre])
            movie_flag = 1.0 if genre in movie_genres else 0.0

            f[f"user_abs_pref_{slug}"] = pref
            f[f"movie_genre_{slug}"] = movie_flag
            f[f"genre_match_{slug}"] = pref * movie_flag

        # 4: relative preference.
        for genre, slug in zip(self.genres, self.genre_slugs):
            f[f"user_rel_pref_{slug}"] = float(genre_rel[genre])

        # 5: evidence = log_count / log1p(200), clipped.
        evidence_den = math.log1p(self.genre_evidence_log_clip)

        for genre, slug in zip(self.genres, self.genre_slugs):
            f[f"user_genre_evidence_{slug}"] = float(
                np.clip(
                    float(genre_log[genre]) / evidence_den,
                    0.0,
                    1.0,
                )
            )

        # 6: known / unknown.
        for genre, slug in zip(self.genres, self.genre_slugs):
            f[f"user_genre_known_{slug}"] = float(genre_known[genre])

        # 7: user scalars.
        f["user_mean_rating_norm"] = self._normalize_rating(
            p["user_mean_rating"]
        )

        history_count = max(0.0, float(p["history_count"]))
        f["user_history_log_count_norm"] = float(
            np.clip(
                math.log1p(history_count)
                / math.log1p(self.user_history_log_clip),
                0.0,
                1.0,
            )
        )

        # 8: metadata affinities.
        affinities = self._affinities(p, movie)
        affinity_values: List[float] = []

        for field in self.affinity_fields:
            value = float(affinities[field])
            f[f"{field}_affinity"] = value
            affinity_values.append(value)

        f["metadata_affinity_mean"] = (
            float(np.mean(affinity_values)) if affinity_values else 0.0
        )
        f["metadata_affinity_max"] = (
            float(np.max(affinity_values)) if affinity_values else 0.0
        )

        # 9: release year, age, runtime.
        release_year, has_year = self._release_year(movie)
        runtime, has_runtime = self._runtime(movie)

        year_value = release_year if has_year else self.year_center
        runtime_value = runtime if has_runtime else self.runtime_center

        f["movie_release_year_scaled"] = float(
            np.clip(
                (year_value - self.year_center) / self.year_scale,
                -3.0,
                2.0,
            )
        )

        age = float(
            np.clip(
                snapshot_year - year_value,
                0.0,
                self.movie_age_max,
            )
        )
        f["movie_age_scaled"] = age / self.movie_age_max

        f["movie_runtime_scaled"] = float(
            np.clip(
                (runtime_value - self.runtime_center) / self.runtime_scale,
                -3.0,
                3.0,
            )
        )

        # 10: metadata presence.
        entity_lists = {
            field: self._entity_ids(movie, field)
            for field in self.affinity_fields
        }

        has_director = float(bool(entity_lists.get("director", [])))
        has_writer = float(bool(entity_lists.get("writer", [])))
        has_cast = float(bool(entity_lists.get("cast", [])))
        has_keyword = float(bool(entity_lists.get("keyword", [])))
        has_company = float(bool(entity_lists.get("company", [])))
        has_collection = float(bool(entity_lists.get("collection", [])))

        f["movie_metadata_available"] = self._metadata_available(
            movie,
            entity_lists=entity_lists,
            has_year=has_year,
            has_runtime=has_runtime,
        )

        # Sama dengan notebook:
        # director + writer + cast + keyword + company + year + runtime.
        f["movie_metadata_completeness"] = float(
            np.mean(
                [
                    has_director,
                    has_writer,
                    has_cast,
                    has_keyword,
                    has_company,
                    float(has_year),
                    float(has_runtime),
                ]
            )
        )

        f["movie_has_collection"] = has_collection
        f["movie_has_year"] = float(has_year)
        f["movie_has_runtime"] = float(has_runtime)

        # 11: metadata amount.
        n_directors = self._entity_count(
            movie, "director", entity_lists["director"]
        )
        n_writers = self._entity_count(
            movie, "writer", entity_lists["writer"]
        )
        n_cast = self._entity_count(
            movie, "cast", entity_lists["cast"]
        )
        n_keywords = self._entity_count(
            movie, "keyword", entity_lists["keyword"]
        )

        f["movie_log_n_directors"] = self._scaled_log_count(
            n_directors, 5.0
        )
        f["movie_log_n_writers"] = self._scaled_log_count(
            n_writers, 15.0
        )
        f["movie_log_n_cast"] = self._scaled_log_count(
            n_cast, 10.0
        )
        f["movie_log_n_keywords"] = self._scaled_log_count(
            n_keywords, 50.0
        )

        # 12: language one-hot.
        language = self._language(movie)

        for lang in self.top_languages:
            f[f"lang_{lang}"] = 1.0 if language == lang else 0.0

        f["lang_other"] = (
            0.0 if language in self.top_languages else 1.0
        )

        self._validate_feature_dict(f)
        return f

    def validate_matrix(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("Feature matrix harus 2D.")

        if matrix.shape[1] != len(self.final_feature_names):
            raise ValueError(
                f"Jumlah feature salah: expected "
                f"{len(self.final_feature_names)}, got {matrix.shape[1]}."
            )

        if matrix.dtype != np.float32:
            raise ValueError("Feature matrix harus dtype float32.")

        if not np.isfinite(matrix).all():
            raise ValueError("Feature matrix mengandung NaN/Inf.")

    def validate_model_file(
        self,
        model_path: Union[str, Path],
    ) -> int:
        """
        Optional runtime contract check.
        Return jumlah feature yang diharapkan XGBoost model.
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "xgboost belum ter-install; tidak bisa validasi model."
            ) from exc

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        model_feature_count = int(booster.num_features())
        expected = len(self.final_feature_names)

        if model_feature_count != expected:
            raise ValueError(
                f"Model mengharapkan {model_feature_count} feature, "
                f"feature_config menghasilkan {expected}."
            )

        return model_feature_count

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def _normalize_profile(self, profile: ProfileLike) -> Dict[str, Any]:
        if isinstance(profile, Mapping):
            data = dict(profile)
        elif hasattr(profile, "to_dict"):
            data = dict(profile.to_dict())
        else:
            attrs = (
                "history_count",
                "user_mean_rating",
                "genre_abs_pref",
                "genre_rel_pref",
                "genre_log_count",
                "genre_known",
                "metadata_profiles",
            )
            if not all(hasattr(profile, a) for a in attrs):
                raise TypeError(
                    "profile harus dict/Mapping atau "
                    "UserPreferenceProfile-compatible."
                )
            data = {a: getattr(profile, a) for a in attrs}

        required = {
            "history_count",
            "user_mean_rating",
            "genre_abs_pref",
            "genre_rel_pref",
            "genre_known",
            "metadata_profiles",
        }

        missing = sorted(required - set(data))
        if missing:
            raise ValueError(
                "Profile kehilangan field: " + ", ".join(missing)
            )

        if "genre_log_count" not in data:
            evidence = data.get("genre_evidence_count")
            if evidence is None:
                raise ValueError(
                    "Profile membutuhkan genre_log_count atau "
                    "genre_evidence_count."
                )
            data["genre_log_count"] = {
                g: math.log1p(max(0.0, float(evidence.get(g, 0.0))))
                for g in self.genres
            }

        for field in (
            "genre_abs_pref",
            "genre_rel_pref",
            "genre_log_count",
            "genre_known",
        ):
            missing_genres = [
                g for g in self.genres if g not in data[field]
            ]
            if missing_genres:
                raise ValueError(
                    f"{field} kehilangan genre: {missing_genres}"
                )

        metadata_profiles = data["metadata_profiles"]

        for field, cfg in self.hash_config.items():
            if field not in metadata_profiles:
                raise ValueError(
                    f"metadata_profiles kehilangan '{field}'."
                )

            vector = np.asarray(
                metadata_profiles[field],
                dtype=np.float32,
            )

            expected_dim = int(cfg["dim"])
            if vector.shape != (expected_dim,):
                raise ValueError(
                    f"metadata_profiles['{field}'] shape harus "
                    f"({expected_dim},), got {vector.shape}."
                )

            if not np.isfinite(vector).all():
                raise ValueError(
                    f"metadata_profiles['{field}'] mengandung NaN/Inf."
                )

        return data

    # ------------------------------------------------------------------
    # Affinity
    # ------------------------------------------------------------------

    def _affinities(
        self,
        profile: Mapping[str, Any],
        movie: MovieRecord,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for field, cfg in self.hash_config.items():
            user_vector = np.asarray(
                profile["metadata_profiles"][field],
                dtype=np.float32,
            )

            norm = float(np.linalg.norm(user_vector))
            if norm > 0.0:
                user_vector = user_vector / norm

            movie_vector = _hashed_movie_vector(
                self._entity_ids(movie, field),
                dim=int(cfg["dim"]),
                salt=str(cfg["salt"]),
            )

            out[field] = float(
                np.clip(
                    np.dot(user_vector, movie_vector),
                    -1.0,
                    1.0,
                )
            )

        return out

    # ------------------------------------------------------------------
    # Movie parsing
    # ------------------------------------------------------------------

    def _movie_id(self, movie: MovieRecord, fallback: Any) -> Any:
        for key in self.MOVIE_ID_ALIASES:
            if key in movie and not _is_missing(movie[key]):
                return movie[key]
        return fallback

    def _genres(self, movie: MovieRecord) -> List[str]:
        raw: List[Any] = []

        for key in self.GENRE_ALIASES:
            if key in movie and not _is_missing(movie[key]):
                raw = _as_list(movie[key])
                break

        result: List[str] = []

        for item in raw:
            if isinstance(item, Mapping):
                item = item.get("name")

            if _is_missing(item):
                continue

            text = str(item).strip()
            if not text or text == "(no genres listed)":
                continue

            lower = text.lower()

            if lower in self.TMDB_GENRE_MAP:
                canonical = self.TMDB_GENRE_MAP[lower]
            else:
                canonical = self.genre_by_lower.get(lower)

            # TMDB "History" dan "TV Movie" tidak punya pasangan langsung
            # di vocabulary MovieLens model final, jadi diabaikan.
            if canonical in self.genres:
                result.append(canonical)

        return list(dict.fromkeys(result))

    def _entity_ids(self, movie: MovieRecord, field: str) -> List[Any]:
        keys = list(self.ENTITY_ALIASES.get(field, ()))
        configured = self.hash_config.get(field, {}).get("column")

        if configured and configured not in keys:
            keys.append(configured)

        for key in keys:
            if key not in movie:
                continue

            result: List[Any] = []

            for value in _as_list(movie[key]):
                if isinstance(value, Mapping):
                    value = value.get("id")

                if _is_missing(value):
                    continue

                if isinstance(value, (float, np.floating)):
                    if float(value).is_integer():
                        value = int(value)

                result.append(value)

            return result

        return []

    def _release_year(self, movie: MovieRecord) -> Tuple[float, bool]:
        for key in ("release_year_clean", "release_year", "year"):
            if key not in movie or _is_missing(movie[key]):
                continue

            try:
                year = float(movie[key])
            except (TypeError, ValueError):
                continue

            if np.isfinite(year) and year > 0:
                return year, True

        if "release_date" in movie and not _is_missing(movie["release_date"]):
            match = re.match(r"^\s*(\d{4})", str(movie["release_date"]))
            if match:
                return float(match.group(1)), True

        if "title" in movie and not _is_missing(movie["title"]):
            match = re.search(r"\((\d{4})\)\s*$", str(movie["title"]))
            if match:
                return float(match.group(1)), True

        return self.year_center, False

    def _runtime(self, movie: MovieRecord) -> Tuple[float, bool]:
        for key in ("runtime_clean", "runtime"):
            if key not in movie or _is_missing(movie[key]):
                continue

            try:
                runtime = float(movie[key])
            except (TypeError, ValueError):
                continue

            # Sama dengan final notebook: <= 0 dianggap missing.
            if np.isfinite(runtime) and runtime > 0:
                return runtime, True

        return self.runtime_center, False

    def _language(self, movie: MovieRecord) -> str:
        for key in (
            "original_language_clean",
            "original_language",
            "language",
        ):
            if key in movie and not _is_missing(movie[key]):
                return str(movie[key]).strip().lower()

        return "unknown"

    def _metadata_available(
        self,
        movie: MovieRecord,
        *,
        entity_lists: Mapping[str, Sequence[Any]],
        has_year: bool,
        has_runtime: bool,
    ) -> float:
        explicit = movie.get("metadata_available")

        if not _is_missing(explicit):
            try:
                return float(bool(int(float(explicit))))
            except (TypeError, ValueError):
                pass

        # Training memakai tmdb_title.notna().
        if not _is_missing(movie.get("tmdb_title")):
            return 1.0

        # Fallback production jika DB tidak menyimpan tmdb_title.
        has_entities = any(bool(v) for v in entity_lists.values())
        has_language = self._language(movie) != "unknown"

        return float(
            has_entities or has_year or has_runtime or has_language
        )

    def _entity_count(
        self,
        movie: MovieRecord,
        field: str,
        fallback_ids: Sequence[Any],
    ) -> float:
        for key in self.COUNT_ALIASES.get(field, ()):
            if key not in movie or _is_missing(movie[key]):
                continue

            try:
                value = float(movie[key])
            except (TypeError, ValueError):
                continue

            if np.isfinite(value):
                return max(0.0, value)

        return float(len(fallback_ids))

    # ------------------------------------------------------------------
    # Scaling + validation
    # ------------------------------------------------------------------

    def _normalize_rating(self, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("user_mean_rating harus numeric.") from exc

        return float(
            np.clip(
                (value - self.rating_min) / self.rating_range,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _scaled_log_count(count: float, reference: float) -> float:
        return float(
            np.clip(
                math.log1p(max(0.0, float(count)))
                / math.log1p(reference),
                0.0,
                1.0,
            )
        )

    def _expected_feature_names(self) -> List[str]:
        names: List[str] = []

        names += [f"user_abs_pref_{s}" for s in self.genre_slugs]
        names += [f"movie_genre_{s}" for s in self.genre_slugs]
        names += [f"genre_match_{s}" for s in self.genre_slugs]
        names += [f"user_rel_pref_{s}" for s in self.genre_slugs]
        names += [f"user_genre_evidence_{s}" for s in self.genre_slugs]
        names += [f"user_genre_known_{s}" for s in self.genre_slugs]

        names += [
            "user_mean_rating_norm",
            "user_history_log_count_norm",
        ]

        names += [
            f"{field}_affinity"
            for field in self.affinity_fields
        ]

        names += [
            "metadata_affinity_mean",
            "metadata_affinity_max",
            "movie_release_year_scaled",
            "movie_age_scaled",
            "movie_runtime_scaled",
            "movie_metadata_available",
            "movie_metadata_completeness",
            "movie_has_collection",
            "movie_has_year",
            "movie_has_runtime",
            "movie_log_n_directors",
            "movie_log_n_writers",
            "movie_log_n_cast",
            "movie_log_n_keywords",
        ]

        names += list(self.language_feature_names)
        return names

    def _validate_config_contract(self) -> None:
        for name, value in (
            ("rating_range", self.rating_range),
            ("year_scale", self.year_scale),
            ("movie_age_max", self.movie_age_max),
            ("runtime_scale", self.runtime_scale),
            ("genre_evidence_log_clip", self.genre_evidence_log_clip),
            ("user_history_log_clip", self.user_history_log_clip),
        ):
            if value <= 0:
                raise ValueError(f"{name} harus > 0.")

        if len(set(self.final_feature_names)) != len(
            self.final_feature_names
        ):
            raise ValueError("final_feature_names mengandung duplikat.")

        forbidden = {
            "movie_historical_rating_norm",
            "movie_historical_log_count_norm",
        }

        bad = sorted(forbidden.intersection(self.final_feature_names))
        if bad:
            raise ValueError(
                "Final config masih mengandung temporal-risk feature: "
                + ", ".join(bad)
            )

        if self.model_config is not None:
            selected = self.model_config.get("selected_model")
            if selected and selected != "leakage_safe_full_content":
                raise ValueError(
                    "feature_builder.py dibuat untuk "
                    "'leakage_safe_full_content', "
                    f"tetapi model_config memilih '{selected}'."
                )

        expected = self._expected_feature_names()

        if set(expected) != set(self.final_feature_names):
            missing = sorted(
                set(self.final_feature_names) - set(expected)
            )
            extra = sorted(
                set(expected) - set(self.final_feature_names)
            )
            raise ValueError(
                "Kontrak feature tidak cocok. "
                f"Missing generated={missing}; extra generated={extra}"
            )

    def _validate_feature_dict(
        self,
        features: Mapping[str, float],
    ) -> None:
        expected = set(self.final_feature_names)
        actual = set(features)

        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Feature mismatch. Missing={missing}; extra={extra}"
            )

        for name in self.final_feature_names:
            if not np.isfinite(float(features[name])):
                raise ValueError(
                    f"Feature '{name}' menghasilkan NaN/Inf."
                )


def load_feature_builder(
    ml_root: Optional[Union[str, Path]] = None,
) -> FeatureBuilder:
    return FeatureBuilder.from_repo_config(ml_root=ml_root)


__all__ = [
    "FeatureBatch",
    "FeatureBuilder",
    "load_feature_builder",
]
