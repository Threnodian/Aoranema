"""
Aoranema - FastAPI Schemas
==========================

Kontrak data HTTP untuk recommendation API.

Tujuan file ini:
- memvalidasi request Laravel sebelum masuk ke ML pipeline,
- menerima `movie_id` maupun `movieId` untuk kompatibilitas,
- memakai format API canonical `snake_case`,
- menjaga metadata film fleksibel karena Laravel dapat mengirim field tambahan
  seperti poster, schedule, cinema, price, dan sebagainya,
- memisahkan request user lama (`history`) dan user baru (`onboarding`),
- memvalidasi response recommendation sebelum dikirim kembali ke Laravel.

Catatan penting:
- `recommendation_score` adalah RAW RANKING SCORE dari XGBRanker,
  BUKAN probability/persentase.
- `movie_id` adalah ID film utama yang dipakai katalog Aoranema/model.
  `tmdb_id` disimpan sebagai metadata terpisah dan tidak menggantikan movie_id.
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------
# Shared primitive types
# ---------------------------------------------------------------------

MovieId = Union[int, str]

MetadataEntity = Union[
    int,
    str,
    Dict[str, Any],
]

MetadataEntityCollection = Union[
    str,  # JSON string from parquet/API, e.g. "[1001, 1002]"
    List[MetadataEntity],
]

GenreCollection = Union[
    str,       # e.g. "Action|Sci-Fi"
    List[str], # e.g. ["Science Fiction", "Family"]
]


def _validate_movie_id_value(
    value: Any,
    *,
    field_name: str = "movie_id",
) -> MovieId:
    """
    IDs may be positive integers or non-empty strings (e.g. UUID/custom ID).
    bool is explicitly rejected because bool is a subclass of int in Python.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} tidak boleh boolean."
        )

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                f"{field_name} integer harus > 0."
            )
        return value

    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            raise ValueError(
                f"{field_name} string tidak boleh kosong."
            )
        return clean

    raise ValueError(
        f"{field_name} harus integer positif atau string non-kosong."
    )


# ---------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------

class ApiBaseModel(BaseModel):
    """
    Default API model:
    - reject unexpected top-level fields,
    - accept field names as declared,
    - reject NaN / +/-Inf,
    - trim normal strings where Pydantic can safely do so.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MovieInput(BaseModel):
    """
    Metadata satu film.

    `extra="allow"` memang disengaja.
    Candidate dari Laravel boleh membawa field business/UI tambahan seperti:
    poster_url, schedules, cinema_name, ticket_price, slug, dll.

    Field tambahan tersebut akan ikut dipertahankan sampai response ranking.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    movie_id: MovieId = Field(
        validation_alias=AliasChoices(
            "movie_id",
            "movieId",
        ),
        description=(
            "ID film utama Aoranema/MovieLens mapping. "
            "Boleh dikirim sebagai movie_id atau movieId."
        ),
    )

    # Optional identification / display metadata.
    tmdb_id: Optional[MovieId] = Field(
        default=None,
        validation_alias=AliasChoices(
            "tmdb_id",
            "tmdbId",
        ),
    )
    title: Optional[str] = None
    tmdb_title: Optional[str] = None
    original_title: Optional[str] = None

    # Genre forms accepted by feature_builder.py.
    genres: Optional[GenreCollection] = None
    movielens_genres: Optional[GenreCollection] = None
    genre_names: Optional[GenreCollection] = None
    genre_names_json: Optional[GenreCollection] = None
    tmdb_genre_names_json: Optional[GenreCollection] = None
    tmdb_genres: Optional[GenreCollection] = None

    # Release/runtime/language aliases used by production FeatureBuilder.
    release_year_clean: Optional[float] = None
    release_year: Optional[float] = None
    year: Optional[float] = None
    release_date: Optional[str] = None

    runtime_clean: Optional[float] = Field(
        default=None,
        ge=0,
    )
    runtime: Optional[float] = Field(
        default=None,
        ge=0,
    )

    original_language_clean: Optional[str] = None
    original_language: Optional[str] = None
    language: Optional[str] = None

    # Hashed metadata fields.
    director_ids_json: Optional[MetadataEntityCollection] = None
    director_ids: Optional[MetadataEntityCollection] = None
    directors: Optional[MetadataEntityCollection] = None

    writer_ids_json: Optional[MetadataEntityCollection] = None
    writer_ids: Optional[MetadataEntityCollection] = None
    writers: Optional[MetadataEntityCollection] = None

    top_cast_ids_json: Optional[MetadataEntityCollection] = None
    top_cast_ids: Optional[MetadataEntityCollection] = None
    cast_ids: Optional[MetadataEntityCollection] = None
    cast: Optional[MetadataEntityCollection] = None

    keyword_ids_json: Optional[MetadataEntityCollection] = None
    keyword_ids: Optional[MetadataEntityCollection] = None
    keywords: Optional[MetadataEntityCollection] = None

    production_company_ids_json: Optional[MetadataEntityCollection] = None
    production_company_ids: Optional[MetadataEntityCollection] = None
    company_ids: Optional[MetadataEntityCollection] = None
    companies: Optional[MetadataEntityCollection] = None

    collection_ids_json: Optional[MetadataEntityCollection] = None
    collection_ids: Optional[MetadataEntityCollection] = None
    collection_id: Optional[MetadataEntity] = None

    # Metadata amount/count features used by final leakage-safe model.
    n_directors: Optional[int] = Field(
        default=None,
        ge=0,
    )
    n_writers: Optional[int] = Field(
        default=None,
        ge=0,
    )
    n_top_cast: Optional[int] = Field(
        default=None,
        ge=0,
    )
    n_cast: Optional[int] = Field(
        default=None,
        ge=0,
    )
    n_keywords: Optional[int] = Field(
        default=None,
        ge=0,
    )

    metadata_available: Optional[
        Union[bool, int, float]
    ] = None

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_movie_id_aliases(
        cls,
        data: Any,
    ) -> Any:
        """
        Avoid ambiguous payloads such as:
            {"movie_id": 10, "movieId": 20}
        """
        if not isinstance(data, Mapping):
            return data

        if (
            "movie_id" in data
            and "movieId" in data
            and data["movie_id"] != data["movieId"]
        ):
            raise ValueError(
                "movie_id dan movieId diberikan bersamaan "
                "dengan nilai yang berbeda."
            )

        return data

    @field_validator(
        "movie_id",
        "tmdb_id",
        mode="before",
    )
    @classmethod
    def validate_ids(
        cls,
        value: Any,
        info,
    ) -> Any:
        if value is None and info.field_name == "tmdb_id":
            return None

        return _validate_movie_id_value(
            value,
            field_name=info.field_name,
        )

    @field_validator(
        "release_year_clean",
        "release_year",
        "year",
    )
    @classmethod
    def validate_year(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None

        # Broad enough for historical/future film data without being arbitrary.
        if not 1800 <= value <= 3000:
            raise ValueError(
                "release year harus berada pada rentang 1800-3000."
            )

        return value

    def to_service_dict(self) -> Dict[str, Any]:
        """
        Convert API representation to the mapping expected by ML builders.

        We intentionally emit `movieId` internally because all production
        components recognize it and it keeps the primary movie ID explicit.
        """
        data = self.model_dump(
            exclude_none=True,
        )

        movie_id = data.pop(
            "movie_id"
        )

        # Never let an allowed extra field override the canonical ID.
        data.pop(
            "movieId",
            None,
        )

        data["movieId"] = movie_id

        return data


class InteractionInput(ApiBaseModel):
    """
    Explicit rating history, on the same 0.5-5.0 scale used in MovieLens.
    """

    movie_id: MovieId = Field(
        validation_alias=AliasChoices(
            "movie_id",
            "movieId",
        ),
    )
    rating: float = Field(
        ge=0.5,
        le=5.0,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_movie_id_aliases(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, Mapping):
            return data

        if (
            "movie_id" in data
            and "movieId" in data
            and data["movie_id"] != data["movieId"]
        ):
            raise ValueError(
                "movie_id dan movieId diberikan bersamaan "
                "dengan nilai yang berbeda."
            )

        return data

    @field_validator(
        "movie_id",
        mode="before",
    )
    @classmethod
    def validate_movie_id(
        cls,
        value: Any,
    ) -> MovieId:
        return _validate_movie_id_value(
            value
        )

    def to_service_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "rating": float(
                self.rating
            ),
        }


# ---------------------------------------------------------------------
# Common recommendation request
# ---------------------------------------------------------------------

class RecommendationRequestBase(ApiBaseModel):
    """
    Fields shared by history and onboarding recommendation requests.
    """

    movie_catalog: List[MovieInput] = Field(
        min_length=1,
        description=(
            "Metadata catalog needed to build the user profile. "
            "Send as a list, not a JSON object keyed by movie ID, so "
            "numeric IDs do not silently become strings in JSON."
        ),
    )

    candidates: List[MovieInput] = Field(
        default_factory=list,
        description=(
            "Films already filtered by Laravel as eligible/current/upcoming "
            "recommendation candidates."
        ),
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return.",
    )

    snapshot_year: Optional[int] = Field(
        default=None,
        ge=1800,
        le=3000,
        description=(
            "Reference year for age-related features. "
            "Later the route may set this explicitly when omitted."
        ),
    )

    min_score: Optional[float] = Field(
        default=None,
        description=(
            "Optional raw XGBRanker score threshold. "
            "This is NOT a probability threshold."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_movie_ids(
        self,
    ):
        catalog_ids = [
            movie.movie_id
            for movie in self.movie_catalog
        ]

        if len(
            catalog_ids
        ) != len(
            set(
                catalog_ids
            )
        ):
            raise ValueError(
                "movie_catalog mengandung movie_id duplikat."
            )

        candidate_ids = [
            movie.movie_id
            for movie in self.candidates
        ]

        if len(
            candidate_ids
        ) != len(
            set(
                candidate_ids
            )
        ):
            raise ValueError(
                "candidates mengandung movie_id duplikat."
            )

        return self

    def movie_catalog_for_service(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            movie.to_service_dict()
            for movie in self.movie_catalog
        ]

    def candidates_for_service(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            movie.to_service_dict()
            for movie in self.candidates
        ]


class HistoryRecommendationRequest(
    RecommendationRequestBase
):
    """
    Recommendation request for a user with explicit ratings/history.
    """

    mode: Literal["history"] = "history"

    interactions: List[
        InteractionInput
    ] = Field(
        min_length=1,
        description="Explicit rating history.",
    )

    def interactions_for_service(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            interaction.to_service_dict()
            for interaction in self.interactions
        ]

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "mode": "history",
                "interactions": [
                    {
                        "movie_id": 1,
                        "rating": 5.0,
                    },
                    {
                        "movie_id": 2,
                        "rating": 2.5,
                    },
                ],
                "movie_catalog": [
                    {
                        "movie_id": 1,
                        "title": "Orbit Protocol",
                        "genres": "Action|Sci-Fi|Thriller",
                        "release_year": 2010,
                        "runtime": 125,
                        "original_language": "en",
                    },
                    {
                        "movie_id": 2,
                        "title": "Summer Letter",
                        "genres": "Drama|Romance",
                        "release_year": 2015,
                        "runtime": 108,
                        "original_language": "fr",
                    },
                ],
                "candidates": [],
                "top_k": 10,
                "snapshot_year": 2026,
            }
        },
    )


class OnboardingRecommendationRequest(
    RecommendationRequestBase
):
    """
    Cold-start request for a new user.
    """

    mode: Literal["onboarding"] = "onboarding"

    favorite_movie_ids: List[
        MovieId
    ] = Field(
        default_factory=list,
        max_length=100,
    )

    favorite_genres: List[str] = Field(
        default_factory=list,
        max_length=50,
    )

    @field_validator(
        "favorite_movie_ids",
        mode="before",
    )
    @classmethod
    def validate_favorite_movie_ids_container(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return []
        return value

    @field_validator(
        "favorite_movie_ids",
    )
    @classmethod
    def validate_favorite_movie_ids(
        cls,
        values: List[MovieId],
    ) -> List[MovieId]:
        cleaned = [
            _validate_movie_id_value(
                value,
                field_name="favorite_movie_ids",
            )
            for value in values
        ]

        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                "favorite_movie_ids mengandung ID duplikat."
            )

        return cleaned

    @field_validator(
        "favorite_genres",
        mode="before",
    )
    @classmethod
    def validate_favorite_genres_container(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return []
        return value

    @field_validator(
        "favorite_genres",
    )
    @classmethod
    def clean_favorite_genres(
        cls,
        values: List[str],
    ) -> List[str]:
        result: List[str] = []

        for value in values:
            clean = value.strip()

            if not clean:
                raise ValueError(
                    "favorite_genres tidak boleh berisi string kosong."
                )

            if clean in result:
                raise ValueError(
                    "favorite_genres mengandung genre duplikat."
                )

            result.append(
                clean
            )

        return result

    @model_validator(mode="after")
    def require_at_least_one_preference_signal(
        self,
    ):
        if (
            not self.favorite_movie_ids
            and not self.favorite_genres
        ):
            raise ValueError(
                "Onboarding membutuhkan minimal satu favorite movie "
                "atau satu favorite genre."
            )

        return self

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
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
                "movie_catalog": [
                    {
                        "movie_id": 1,
                        "title": "Orbit Protocol",
                        "genres": "Action|Sci-Fi|Thriller",
                    },
                    {
                        "movie_id": 5,
                        "title": "Animated Galaxy",
                        "genres": "Adventure|Animation|Children|Sci-Fi",
                    },
                ],
                "candidates": [],
                "top_k": 10,
                "snapshot_year": 2026,
            }
        },
    )


# Discriminated union for a single POST /recommendations endpoint.
RecommendationRequest = Annotated[
    Union[
        HistoryRecommendationRequest,
        OnboardingRecommendationRequest,
    ],
    Field(
        discriminator="mode"
    ),
]

RecommendationRequestAdapter = TypeAdapter(
    RecommendationRequest
)


# ---------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------

class RecommendationItem(BaseModel):
    """
    Ranked movie returned to Laravel.

    Extra candidate fields are retained.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    movie_id: MovieId = Field(
        validation_alias=AliasChoices(
            "movie_id",
            "movieId",
        ),
    )
    rank: int = Field(
        ge=1,
    )
    recommendation_score: float = Field(
        description=(
            "Raw XGBRanker score; higher is better. "
            "Not a probability."
        )
    )

    @field_validator(
        "movie_id",
        mode="before",
    )
    @classmethod
    def validate_movie_id(
        cls,
        value: Any,
    ) -> MovieId:
        return _validate_movie_id_value(
            value
        )


class RecommendationResponse(ApiBaseModel):
    recommendations: List[
        RecommendationItem
    ]
    candidate_count: int = Field(
        ge=0,
    )
    returned_count: int = Field(
        ge=0,
    )
    model_name: str
    profile_source: str
    warnings: List[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_counts_and_ranks(
        self,
    ):
        if (
            self.returned_count
            != len(
                self.recommendations
            )
        ):
            raise ValueError(
                "returned_count tidak sama dengan jumlah recommendations."
            )

        if (
            self.returned_count
            > self.candidate_count
        ):
            raise ValueError(
                "returned_count tidak boleh lebih besar dari candidate_count."
            )

        expected_ranks = list(
            range(
                1,
                len(
                    self.recommendations
                )
                + 1,
            )
        )

        actual_ranks = [
            item.rank
            for item in self.recommendations
        ]

        if actual_ranks != expected_ranks:
            raise ValueError(
                "Rank response harus berurutan mulai dari 1."
            )

        return self

    @classmethod
    def from_service_result(
        cls,
        result: Any,
    ) -> "RecommendationResponse":
        return cls(
            recommendations=[
                RecommendationItem.model_validate(
                    item
                )
                for item
                in result.recommendations
            ],
            candidate_count=(
                result.candidate_count
            ),
            returned_count=(
                result.returned_count
            ),
            model_name=(
                result.model_name
            ),
            profile_source=(
                result.profile_source
            ),
            warnings=(
                result.warnings
                or []
            ),
        )


# ---------------------------------------------------------------------
# Health / metadata schemas
# ---------------------------------------------------------------------

class ModelInfoResponse(ApiBaseModel):
    model_name: str
    model_file: str
    feature_count: int = Field(
        ge=1,
    )
    model_feature_count: int = Field(
        ge=1,
    )
    total_boosted_rounds: int = Field(
        ge=1,
    )
    best_iteration: Optional[int] = Field(
        default=None,
        ge=0,
    )
    best_score: Optional[float] = None
    scores_are_probabilities: bool
    ranking_direction: Literal[
        "higher_score_is_better"
    ]


class ServiceInfoResponse(ApiBaseModel):
    service: str
    model: ModelInfoResponse
    default_top_k: int = Field(
        ge=1,
    )
    pipeline: List[str]

    @classmethod
    def from_service_info(
        cls,
        info: Mapping[str, Any],
    ) -> "ServiceInfoResponse":
        return cls.model_validate(
            info
        )


class HealthResponse(ApiBaseModel):
    status: Literal["ok"] = "ok"
    service: ServiceInfoResponse


__all__ = [
    "MovieId",
    "MovieInput",
    "InteractionInput",
    "HistoryRecommendationRequest",
    "OnboardingRecommendationRequest",
    "RecommendationRequest",
    "RecommendationRequestAdapter",
    "RecommendationItem",
    "RecommendationResponse",
    "ModelInfoResponse",
    "ServiceInfoResponse",
    "HealthResponse",
]
