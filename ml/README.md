# Aoranema ML Recommendation Service

Repository ini berisi komponen **machine learning recommendation system** untuk Aoranema. Model akhir menggunakan **XGBRanker** dan dijalankan melalui **FastAPI** sebagai service terpisah yang nantinya dipanggil oleh backend Laravel.

## Status

Pipeline production saat ini:

```text
history / onboarding
        ↓
PreferenceBuilder
        ↓
FeatureBuilder (149 features)
        ↓
RecommendationPredictor
        ↓
MovieRanker
        ↓
Top-N recommendations
        ↓
FastAPI
```

Model production yang digunakan:

```text
leakage_safe_full_content
```

`recommendation_score` adalah **raw ranking score**, bukan probabilitas atau persentase. Nilai yang lebih besar berarti kandidat ditempatkan lebih tinggi.

## Struktur Repository

```text
ml/
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       └── recommendation.py
│
├── config/
│   ├── feature_config.json
│   ├── model_config.json
│   ├── preference_config.json
│   └── audit_report.json
│
├── models/
│   └── final_recommender_xgboost_ranker.json
│
├── notebooks/
│   ├── 01_baseline_recommendation_v1.ipynb
│   ├── 02_tmdb_metadata_collection.ipynb
│   └── 03_advanced_recommendation_v2_final_audit.ipynb
│
├── results/
│   └── ...
│
├── src/
│   ├── preference_builder.py
│   ├── feature_builder.py
│   ├── predictor.py
│   ├── ranker.py
│   └── recommendation_service.py
│
├── tests/
│   ├── test_recommendation_pipeline.py
│   ├── test_recommendation_service.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Dataset besar tidak disimpan di Git. `data/raw/` dan `data/processed/` bersifat lokal.

## Setup Development

Dari folder `ml`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Jika virtual environment sudah ada, cukup aktifkan dan install dependency:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Menjalankan Test

Jalankan seluruh test:

```powershell
python -m pytest -q tests
```

Pipeline saat ini memiliki tiga kelompok integration test:

```text
test_recommendation_pipeline.py
test_recommendation_service.py
test_api.py
```

Semua test harus lolos sebelum perubahan source digabungkan.

## Menjalankan FastAPI

Dari folder `ml`:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Setelah server hidup:

```text
API          http://127.0.0.1:8001
Health       http://127.0.0.1:8001/health
Swagger      http://127.0.0.1:8001/docs
ReDoc        http://127.0.0.1:8001/redoc
OpenAPI      http://127.0.0.1:8001/openapi.json
```

`GET /health` memastikan service dan model production berhasil dimuat.

## Endpoint Recommendation

```text
POST /recommendations
```

### Existing User

Contoh ringkas:

```json
{
  "mode": "history",
  "interactions": [
    {"movie_id": 1, "rating": 5.0},
    {"movie_id": 2, "rating": 2.5}
  ],
  "movie_catalog": [
    {
      "movie_id": 1,
      "genres": "Action|Sci-Fi",
      "release_year": 2010
    },
    {
      "movie_id": 2,
      "genres": "Drama|Romance",
      "release_year": 2015
    }
  ],
  "candidates": [],
  "top_k": 10,
  "snapshot_year": 2026
}
```

### New User / Cold Start

```json
{
  "mode": "onboarding",
  "favorite_movie_ids": [1, 5],
  "favorite_genres": ["Action", "Sci-Fi"],
  "movie_catalog": [
    {
      "movie_id": 1,
      "genres": "Action|Sci-Fi"
    },
    {
      "movie_id": 5,
      "genres": "Adventure|Animation|Children|Sci-Fi"
    }
  ],
  "candidates": [],
  "top_k": 10,
  "snapshot_year": 2026
}
```

Dalam penggunaan nyata, `candidates` diisi film yang sudah difilter backend sebagai film yang layak direkomendasikan, misalnya film yang sedang tayang atau upcoming.

## Pembagian Tanggung Jawab

### Laravel

Laravel menjadi sumber data aplikasi dan bertanggung jawab atas:

- user dan autentikasi;
- rating/history user;
- film yang sedang tayang atau upcoming;
- jadwal bioskop;
- harga tiket;
- poster dan detail tampilan;
- filtering kandidat yang memang tersedia.

### ML API

Service ini bertanggung jawab atas:

- membuat user preference;
- membangun 149 feature production;
- melakukan inference XGBRanker;
- mengurutkan candidate;
- mengembalikan Top-N recommendation.

Arsitektur:

```text
Laravel
   ↓ HTTP
FastAPI ML Service
   ↓
XGBRanker
   ↓
ranked movie IDs / candidate metadata
```

Kode model tidak dipindahkan ke Laravel.

## CORS

Laravel yang memanggil FastAPI dari backend tidak membutuhkan CORS.

Jika browser/frontend perlu memanggil FastAPI secara langsung, buat `.env` atau set environment variable:

```powershell
$env:AORANEMA_CORS_ORIGINS="http://localhost:5173,http://localhost:8000"
```

Jangan gunakan wildcard origin untuk deployment tanpa alasan yang jelas.

## Data dan Model

Dataset training besar tidak dimasukkan ke repository Git.

File production yang harus tersedia:

```text
models/final_recommender_xgboost_ranker.json
config/feature_config.json
config/model_config.json
config/preference_config.json
```

Model dan feature config harus tetap sinkron. Predictor akan menolak inference jika jumlah feature tidak cocok.

## Evaluasi

Model final dipilih berdasarkan validation ranking evaluation menggunakan protokol sampled-candidate yang leakage-safe.

Saat menjelaskan hasil model, gunakan istilah metric yang benar seperti **NDCG@10**, **Precision@10**, **Recall@10**, **HitRate@10**, dan **MAP@10**.

Jangan menyebut NDCG sebagai accuracy.

## Workflow Perubahan Kode

Sebelum commit perubahan di `src/` atau `api/`:

```powershell
python -m pytest -q tests
```

Jika ada test gagal, perbaiki dulu sebelum integrasi ke Laravel.
