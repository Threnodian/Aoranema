# Aoranema ML Recommendation Service

Service machine learning untuk fitur **personalized movie recommendation** pada Aoranema.

Repository ini bukan sekadar berisi file model. Repository ini berisi **pipeline lengkap** yang mengubah histori/preferensi user menjadi ranking film, lalu membungkusnya sebagai **HTTP API menggunakan FastAPI** agar backend Laravel dapat memakainya.

---

# 1. Gambaran Paling Sederhana

Jika masih awam, bayangkan pembagian sistem Aoranema seperti ini:

```text
User membuka Aoranema
        ↓
Frontend
        ↓
Laravel Backend
        ↓
Laravel mengumpulkan:
- histori/rating user
- pilihan onboarding
- film yang sedang tayang/upcoming
- metadata film
        ↓
Laravel mengirim JSON ke ML API
        ↓
FastAPI Recommendation Service
        ↓
PreferenceBuilder
        ↓
FeatureBuilder
        ↓
XGBRanker
        ↓
MovieRanker
        ↓
Top-N recommendation
        ↓
JSON dikembalikan ke Laravel
        ↓
Laravel menampilkan hasil ke frontend
```

## Apakah cukup memberikan satu link model lalu backend langsung bisa memakai ML?

**Tidak.**

File berikut:

```text
models/final_recommender_xgboost_ranker.json
```

hanyalah file model XGBoost. Backend Laravel tidak dapat langsung mengirim user ke file tersebut lalu mendapatkan rekomendasi.

Model membutuhkan:

```text
preference user
+
metadata film
+
149 feature dalam urutan yang tepat
+
proses inference
+
proses ranking
```

Karena itu repository ini menyediakan FastAPI sebagai lapisan penghubung.

Setelah FastAPI dijalankan atau di-deploy, **barulah backend Laravel cukup mengenal URL API**, misalnya:

```text
http://127.0.0.1:8001
```

untuk development lokal, atau misalnya:

```text
https://ml.aoranema.example
```

setelah service benar-benar di-deploy ke server.

Jadi konsep yang benar adalah:

```text
BUKAN:

Laravel
→ link file model.json
→ langsung rekomendasi


TETAPI:

Laravel
→ HTTP request
→ FastAPI ML Service
→ model + feature pipeline
→ HTTP response
→ Laravel
```

---

# 2. Model yang Digunakan

Model production:

```text
leakage_safe_full_content
```

Jenis model:

```text
XGBRanker
```

Jumlah feature:

```text
149
```

Model melakukan **ranking**, bukan klasifikasi probabilitas.

Karena itu nilai seperti:

```text
recommendation_score = -1.5
```

bukan berarti:

```text
-150%
```

dan nilai:

```text
0.8
```

bukan berarti:

```text
80% cocok
```

Yang penting adalah **perbandingan score**:

```text
Film A = -1.2
Film B = -2.7

-1.2 > -2.7

maka Film A diranking lebih tinggi.
```

Backend sebaiknya menggunakan:

```text
rank
```

yang sudah dikembalikan API, bukan membuat threshold sendiri seperti:

```php
if ($score > 0.5) ...
```

---

# 3. Struktur Repository

```text
ml/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       └── recommendation.py
│
├── config/
│   ├── audit_report.json
│   ├── feature_config.json
│   ├── model_config.json
│   └── preference_config.json
│
├── data/
│   └── raw/
│       └── ml-32m/
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
│   ├── ablation_validation.csv
│   ├── data_quality.json
│   ├── demo_top10.csv
│   ├── final_feature_importance.csv
│   ├── history_cohort_metrics.csv
│   ├── metrics_v2_final_audit.json
│   ├── test_audit_comparison.csv
│   └── validation_audit_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── preference_builder.py
│   ├── feature_builder.py
│   ├── predictor.py
│   ├── ranker.py
│   └── recommendation_service.py
│
├── tests/
│   ├── test_api.py
│   ├── test_recommendation_pipeline.py
│   └── test_recommendation_service.py
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

# 4. Fungsi Setiap Bagian

## `src/preference_builder.py`

Mengubah histori rating atau hasil onboarding menjadi profil preferensi user.

Contoh input:

```text
User memberi rating tinggi pada Action dan Sci-Fi
```

menjadi informasi numerik seperti:

```text
seberapa kuat user menyukai Action
seberapa kuat user menyukai Sci-Fi
berapa banyak bukti historinya
preferensi terhadap director/cast/writer/keyword/company/collection
```

## `src/feature_builder.py`

Menggabungkan:

```text
profil user
+
metadata candidate movie
```

menjadi **149 feature** dalam format yang sama seperti ketika model dilatih.

Urutan feature tidak boleh berubah.

## `src/predictor.py`

Memuat:

```text
models/final_recommender_xgboost_ranker.json
```

lalu menghasilkan raw ranking score untuk setiap candidate.

## `src/ranker.py`

Mengurutkan score dari terbesar ke terkecil dan menghasilkan:

```text
rank 1
rank 2
rank 3
...
```

## `src/recommendation_service.py`

Menyatukan:

```text
PreferenceBuilder
→ FeatureBuilder
→ Predictor
→ Ranker
```

agar layer API cukup melakukan satu pemanggilan service.

## `api/schemas.py`

Menentukan bentuk JSON yang boleh masuk dan keluar.

Pydantic akan menolak request yang tidak valid sebelum data masuk terlalu jauh ke pipeline ML.

## `api/routes/recommendation.py`

Menyediakan endpoint:

```text
POST /recommendations
```

## `api/main.py`

Entry point FastAPI.

Menyediakan:

```text
GET  /
GET  /health
POST /recommendations
GET  /docs
GET  /redoc
GET  /openapi.json
```

---

# 5. Setup dari Nol

Requirement utama:

```text
Python
pip
```

Disarankan memakai virtual environment.

Dari folder:

```text
C:\Aoranema\ml
```

buat virtual environment:

```powershell
python -m venv .venv
```

Aktifkan di PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Untuk development dan testing:

```powershell
python -m pip install -r requirements-dev.txt
```

Untuk deployment production yang tidak menjalankan test:

```powershell
python -m pip install -r requirements.txt
```

---

# 6. Menjalankan Semua Test

Dari folder `ml`:

```powershell
python -m pytest -q tests
```

Saat dokumentasi ini dibuat, suite terdiri dari:

```text
test_recommendation_pipeline.py
test_recommendation_service.py
test_api.py
```

Semua test harus lolos sebelum source code production diubah atau diintegrasikan ke backend.

---

# 7. Menjalankan ML API

Dari:

```text
C:\Aoranema\ml
```

jalankan:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Jika berhasil:

```text
Uvicorn running on http://127.0.0.1:8001
Application startup complete.
```

---

# 8. Endpoint Penting

## Base URL lokal

```text
http://127.0.0.1:8001
```

## Health check

```text
GET /health
```

Contoh:

```text
http://127.0.0.1:8001/health
```

Tujuan:

```text
cek FastAPI hidup
+
cek RecommendationService berhasil dibuat
+
cek model berhasil dimuat
+
cek jumlah feature cocok
```

Response normal berisi informasi seperti:

```json
{
  "status": "ok",
  "service": {
    "service": "aoranema_movie_recommendation",
    "model": {
      "model_name": "leakage_safe_full_content",
      "feature_count": 149,
      "model_feature_count": 149,
      "scores_are_probabilities": false,
      "ranking_direction": "higher_score_is_better"
    },
    "default_top_k": 10
  }
}
```

## Swagger

```text
GET /docs
```

Buka:

```text
http://127.0.0.1:8001/docs
```

Swagger dapat digunakan untuk mencoba API secara manual tanpa Laravel.

---

# 9. Endpoint Recommendation

```text
POST /recommendations
```

API mendukung dua mode:

```text
history
onboarding
```

---

# 10. Mode `history`

Dipakai untuk user yang sudah mempunyai explicit rating/history.

Contoh request:

```json
{
  "mode": "history",
  "interactions": [
    {
      "movie_id": 1,
      "rating": 5.0
    },
    {
      "movie_id": 2,
      "rating": 2.5
    }
  ],
  "movie_catalog": [
    {
      "movie_id": 1,
      "title": "Orbit Protocol",
      "genres": "Action|Sci-Fi|Thriller",
      "release_year": 2010,
      "runtime": 125,
      "original_language": "en"
    },
    {
      "movie_id": 2,
      "title": "Summer Letter",
      "genres": "Drama|Romance",
      "release_year": 2015,
      "runtime": 108,
      "original_language": "fr"
    }
  ],
  "candidates": [
    {
      "movie_id": 3,
      "title": "Deep Frontier",
      "genres": "Action|Adventure|Sci-Fi",
      "release_year": 2022,
      "runtime": 135,
      "original_language": "en"
    }
  ],
  "top_k": 10,
  "snapshot_year": 2026
}
```

---

# 11. Mode `onboarding`

Dipakai untuk user baru yang belum punya rating history.

Contoh:

```json
{
  "mode": "onboarding",
  "favorite_movie_ids": [1, 5],
  "favorite_genres": [
    "Action",
    "Sci-Fi",
    "Adventure"
  ],
  "movie_catalog": [
    {
      "movie_id": 1,
      "title": "Orbit Protocol",
      "genres": "Action|Sci-Fi|Thriller"
    },
    {
      "movie_id": 5,
      "title": "Animated Galaxy",
      "genres": "Adventure|Animation|Children|Sci-Fi"
    }
  ],
  "candidates": [
    {
      "movie_id": 3,
      "title": "Deep Frontier",
      "genres": "Action|Adventure|Sci-Fi"
    }
  ],
  "top_k": 10,
  "snapshot_year": 2026
}
```

Onboarding membutuhkan minimal:

```text
favorite_movie_ids
atau
favorite_genres
```

Salah satu boleh kosong, tetapi keduanya tidak boleh kosong bersamaan.

---

# 12. Arti `interactions`

Contoh:

```json
{
  "movie_id": 10,
  "rating": 4.5
}
```

Artinya user memberi rating 4.5 pada `movie_id` 10.

Rating API menggunakan skala:

```text
0.5 - 5.0
```

---

# 13. Perbedaan `movie_catalog` dan `candidates`

Bagian ini **sangat penting untuk backend**.

## `movie_catalog`

Digunakan untuk membaca metadata film yang muncul pada histori atau pilihan onboarding user.

Contoh:

```text
User pernah rating:
movie 10
movie 20
movie 30
```

Maka Laravel harus memberikan metadata untuk film tersebut pada:

```text
movie_catalog
```

Kalau histori mereferensikan:

```text
movie_id = 20
```

tetapi movie 20 tidak ada di `movie_catalog`, API akan menganggap payload tidak lengkap.

## `candidates`

Berisi film yang **saat ini boleh dipertimbangkan sebagai rekomendasi**.

Contoh:

```text
film yang sedang tayang
film upcoming
film yang tersedia di Aoranema
```

Jadi:

```text
movie_catalog
= membantu memahami user

candidates
= film yang akan diranking
```

Contoh konseptual:

```text
User pernah suka:
Interstellar
Inception
The Dark Knight

movie_catalog:
Interstellar
Inception
The Dark Knight

candidate film sekarang:
Film A
Film B
Film C
Film D

ML:
profil user
+
Film A-D
↓
ranking Film A-D
```

---

# 14. Siapa yang Menentukan Film Sedang Tayang?

**Laravel/backend.**

ML API tidak mengetahui:

```text
jadwal bioskop hari ini
kursi tersedia
harga tiket
film sudah turun layar
film belum dirilis
cabang bioskop
```

Karena itu backend harus melakukan filtering terlebih dahulu.

```text
semua film database
↓
Laravel filter
↓
currently showing / upcoming
↓
candidates
↓
ML
```

---

# 15. Metadata Film

Model bisa memakai metadata seperti:

```text
genres
director
writer
cast
keywords
production company
collection
release year
runtime
language
```

Contoh field yang didukung antara lain:

```text
genres
release_year
runtime
original_language
director_ids_json
writer_ids_json
top_cast_ids_json
keyword_ids_json
production_company_ids_json
collection_ids_json
n_directors
n_writers
n_top_cast
n_keywords
```

Tidak semua field harus selalu tersedia agar API dapat menerima film.

Namun metadata yang lebih lengkap memberi model lebih banyak informasi untuk ranking.

---

# 16. Field Tambahan dari Laravel

Candidate boleh membawa field tambahan seperti:

```text
poster_url
ticket_price
cinema_name
schedule
slug
```

Contoh:

```json
{
  "movie_id": 3,
  "title": "Deep Frontier",
  "genres": "Action|Sci-Fi",
  "poster_url": "/poster/deep-frontier.jpg",
  "ticket_price": 50000,
  "cinema_name": "Aoranema Central"
}
```

ML tidak harus menggunakan semua field tersebut.

Field tambahan candidate tetap dipertahankan sehingga dapat kembali bersama response ranking.

---

# 17. Response Recommendation

Contoh:

```json
{
  "recommendations": [
    {
      "movie_id": 3,
      "title": "Deep Frontier",
      "poster_url": "/poster/deep-frontier.jpg",
      "ticket_price": 50000,
      "rank": 1,
      "recommendation_score": -1.669089
    },
    {
      "movie_id": 4,
      "title": "Laugh Track",
      "poster_url": "/poster/laugh-track.jpg",
      "ticket_price": 45000,
      "rank": 2,
      "recommendation_score": -2.849372
    }
  ],
  "candidate_count": 2,
  "returned_count": 2,
  "model_name": "leakage_safe_full_content",
  "profile_source": "history",
  "warnings": []
}
```

---

# 18. Arti Response

## `recommendations`

Film yang sudah diurutkan.

## `rank`

```text
1 = recommendation tertinggi
2 = berikutnya
3 = berikutnya
```

Backend/frontend paling aman menggunakan field ini untuk urutan tampilan.

## `recommendation_score`

Raw score XGBRanker.

```text
semakin tinggi
→ semakin tinggi ranking
```

Bukan probability.

## `candidate_count`

Jumlah film yang dipertimbangkan model.

## `returned_count`

Jumlah film yang dikembalikan.

## `profile_source`

Contoh:

```text
history
```

atau:

```text
onboarding
```

## `warnings`

Warning non-fatal jika ada.

Normalnya:

```json
[]
```

---

# 19. ID Film

Gunakan:

```text
movie_id
```

secara konsisten di integrasi Laravel.

API juga menerima alias `movieId` di beberapa input, tetapi output dinormalkan menjadi `movie_id`.

---

# 20. `tmdb_id` Bukan Pengganti `movie_id`

Jika Aoranema memiliki ID internal:

```text
movie_id
```

gunakan itu sebagai ID utama.

`tmdb_id` adalah metadata tambahan.

Jangan tiba-tiba mengganti arti `movie_id` menjadi TMDB ID tanpa mapping yang konsisten.

---

# 21. `top_k`

Default:

```text
10
```

Batas schema:

```text
1 - 100
```

Jika candidate hanya 4 film dan `top_k = 10`, response tetap hanya 4 film.

---

# 22. `snapshot_year`

Dipakai untuk feature umur film.

Untuk integration awal, backend disarankan mengirim tahun saat recommendation dibuat.

```json
"snapshot_year": 2026
```

---

# 23. `min_score`

Opsional.

Untuk MVP Aoranema, sebaiknya tidak digunakan kecuali ada alasan yang sudah diuji.

Jangan menganggap:

```text
min_score = 0.5
```

sebagai threshold 50%.

Score bukan probability.

---

# 24. HTTP Status

## `200 OK`

Recommendation berhasil.

## `422`

Input tidak valid atau tidak lengkap.

Contoh:

```text
rating di luar skala
top_k tidak valid
history movie tidak ada di movie_catalog
duplicate candidate
onboarding tanpa preference signal
```

## `500`

Masalah internal ML service.

## Connection refused / timeout

Biasanya:

```text
FastAPI tidak hidup
URL salah
port salah
network/container tidak terhubung
```

---

# 25. Contoh Integrasi Laravel

`.env` Laravel:

```env
ML_API_URL=http://127.0.0.1:8001
```

`config/services.php`:

```php
'ml' => [
    'url' => env('ML_API_URL', 'http://127.0.0.1:8001'),
],
```

Health check:

```php
use Illuminate\Support\Facades\Http;

$response = Http::timeout(5)
    ->get(config('services.ml.url') . '/health');

if ($response->successful()) {
    $data = $response->json();
}
```

Recommendation:

```php
$response = Http::timeout(15)
    ->post(
        config('services.ml.url') . '/recommendations',
        [
            'mode' => 'history',

            'interactions' => [
                [
                    'movie_id' => 1,
                    'rating' => 5.0,
                ],
                [
                    'movie_id' => 2,
                    'rating' => 2.5,
                ],
            ],

            'movie_catalog' => $movieCatalog,

            'candidates' => $candidates,

            'top_k' => 10,

            'snapshot_year' => now()->year,
        ]
    );

if ($response->successful()) {
    $recommendations = $response->json('recommendations');
}
```

Error handling:

```php
if ($response->successful()) {
    return $response->json();
}

if ($response->status() === 422) {
    // Payload backend -> ML tidak valid.
}

if ($response->serverError()) {
    // ML service gagal.
}
```

Production code sebaiknya juga memiliki timeout, logging, dan fallback.

---

# 26. Jangan Memanggil ML API Satu Kali per Film

Cara salah:

```text
Film A → request API
Film B → request API
Film C → request API
```

Cara benar:

```text
Laravel kumpulkan [A, B, C]
↓
satu POST /recommendations
↓
ML ranking semuanya
```

---

# 27. Jangan Retrain Model Setiap Ada User Baru

Model global tidak perlu dilatih ulang setiap user login atau memberi rating.

```text
history user berubah
↓
preference profile berubah
↓
feature berubah
↓
model yang sama
↓
ranking baru
```

---

# 28. Rating, Wishlist, Booking, dan Klik

Model saat ini dibangun terutama dari explicit rating MovieLens.

Karena itu jangan otomatis menyamakan:

```text
wishlist = rating 4.5
booking = rating 5
click = rating 4
```

tanpa desain dan pengujian tambahan.

---

# 29. Cold Start User Baru

User baru belum memiliki history.

Gunakan mode:

```text
onboarding
```

dengan:

```text
favorite genres
favorite movies
```

Setelah user punya explicit rating/history yang memadai, backend dapat menggunakan mode:

```text
history
```

---

# 30. Siapa yang Menyimpan Data User?

Laravel tetap menjadi sumber data aplikasi:

```text
user
auth
ratings
onboarding choices
movie availability
schedule
price
```

ML API tidak menggantikan database backend.

---

# 31. CORS

Jika:

```text
Laravel backend → FastAPI
```

komunikasi server-to-server, CORS tidak diperlukan.

Jika browser frontend memanggil FastAPI secara langsung, gunakan:

```text
AORANEMA_CORS_ORIGINS
```

Contoh:

```powershell
$env:AORANEMA_CORS_ORIGINS="http://localhost:5173,http://localhost:8000"
```

---

# 32. Localhost Penting Dipahami

```text
127.0.0.1
```

berarti mesin tempat request dibuat.

Jika Laravel dan FastAPI ada pada mesin berbeda, `127.0.0.1` tidak bisa dipakai untuk menunjuk server lainnya.

Gunakan hostname, private IP, service name container, atau domain sesuai deployment.

---

# 33. Apa Artinya "Kasih Link ML ke Backend"?

Setelah deployment, backend memang dapat cukup menyimpan:

```env
ML_API_URL=https://ml-api.aoranema.example
```

Tetapi URL tersebut bekerja karena di belakangnya ada:

```text
server
Python
FastAPI
source pipeline
config
model final
```

Jadi URL itu adalah alamat **service yang hidup**, bukan link download model.

---

# 34. Deployment Production

Secara konsep:

```text
Git repository
↓
server/container
↓
install requirements.txt
↓
model + config tersedia
↓
start FastAPI
↓
expose endpoint
↓
Laravel memanggil endpoint
```

Deployment dapat menggunakan VM/VPS, container, Docker, atau platform cloud yang mendukung Python web service.

---

# 35. Security Production

Jika ML API dibuka keluar network, pertimbangkan:

```text
HTTPS
network restriction
service authentication/token
rate limiting
request size limits
logging
monitoring
secret management
```

Untuk integration lokal awal, semua ini belum harus diterapkan sekaligus.

---

# 36. Health Check dan Fallback Backend

Backend dapat melakukan:

```text
GET /health
```

Jika ML unavailable, sebaiknya homepage tidak ikut crash.

Contoh fallback:

```text
ML gagal
↓
gunakan film populer / terbaru
↓
website tetap berfungsi
```

---

# 37. Data Besar Tidak Masuk Git

Dataset training seperti MovieLens tidak perlu masuk repository Git.

```text
data/raw/
```

bersifat lokal.

Model production:

```text
models/final_recommender_xgboost_ranker.json
```

tetap diperlukan oleh inference service.

---

# 38. Notebook Bukan Runtime Production

Backend **tidak perlu menjalankan notebook**.

Folder:

```text
notebooks/
```

dipakai untuk eksperimen, training, metadata collection, dan audit.

Runtime recommendation memakai:

```text
src/
api/
config/
models/
```

---

# 39. Config Production

File penting:

```text
config/feature_config.json
config/model_config.json
config/preference_config.json
models/final_recommender_xgboost_ranker.json
```

Model dan feature config harus tetap sinkron.

---

# 40. Testing

```text
test_recommendation_pipeline.py
→ ML pipeline

test_recommendation_service.py
→ orchestration

test_api.py
→ HTTP API
```

Run:

```powershell
python -m pytest -q tests
```

---

# 41. Evaluasi Model

Final model menggunakan ranking metrics seperti:

```text
Precision@10
Recall@10
NDCG@10
HitRate@10
MAP@10
```

Pada sampled candidate ranking test yang leakage-safe, final model mencapai:

```text
NDCG@10 ≈ 0.692
```

Ini **bukan accuracy 69.2%**.

---

# 42. Training vs Production

Training:

```text
dataset
↓
eksperimen
↓
validation
↓
test
↓
model final
```

Production:

```text
model final
↓
request user
↓
feature
↓
inference
↓
ranking
```

Backend tidak menjalankan training saat aplikasi dipakai.

---

# 43. Flow Backend yang Direkomendasikan

User lama:

```text
1. User membuka halaman.
2. Laravel ambil rating/history user.
3. Laravel ambil metadata film history.
4. Laravel ambil film eligible/current/upcoming.
5. Laravel bentuk payload.
6. Laravel POST /recommendations.
7. FastAPI mengembalikan ranking.
8. Laravel menampilkan Top-N.
```

User baru:

```text
1. User selesai onboarding.
2. Laravel punya favorite genres/movies.
3. Laravel ambil metadata favorite movies.
4. Laravel ambil candidate eligible.
5. Laravel POST mode=onboarding.
6. FastAPI mengembalikan ranking.
7. Laravel menampilkan hasil.
```

---

# 44. Checklist Integrasi Backend

```text
[ ] ML API dapat dijalankan
[ ] GET /health = 200
[ ] Laravel memiliki ML_API_URL
[ ] Laravel dapat GET /health
[ ] Laravel dapat POST /recommendations
[ ] movie_id mapping konsisten
[ ] history movie ada di movie_catalog
[ ] favorite movie ada di movie_catalog
[ ] candidates hanya film eligible
[ ] top_k sesuai UI
[ ] 422 ditangani
[ ] 500 ditangani
[ ] timeout/connection failure ditangani
[ ] response rank dipakai dengan benar
[ ] score tidak dianggap probability
[ ] fallback UI tersedia jika ML down
[ ] end-to-end test dilakukan
```

---

# 45. FAQ

## Backend cukup dikasih file model?

Tidak.

Backend membutuhkan service yang menjalankan preprocessing, feature building, inference, dan ranking.

## Backend cukup dikasih URL?

Setelah ML service sudah di-deploy dan hidup, **ya secara operasional backend cukup memakai base URL API**.

Tetapi URL itu adalah URL FastAPI service, bukan URL file model.

## Laravel harus install XGBoost?

Tidak.

XGBoost hanya berada di service Python/ML.

## Laravel harus tahu 149 feature?

Tidak.

Laravel menyediakan data user dan film. Python yang membangun 149 feature.

## Laravel harus mengurutkan score?

Tidak perlu. Gunakan `rank` dari response.

## Score negatif berarti buruk?

Tidak.

Score adalah ranking score relatif.

## Model ditraining ulang setiap user?

Tidak.

## Notebook harus dijalankan di server?

Tidak.

## Kalau candidate kosong?

API dapat mengembalikan recommendation kosong dengan status sukses.

## Kalau ML mati apakah seluruh website harus mati?

Tidak seharusnya. Backend idealnya punya fallback.

---

# 46. Batas Scope Saat Ini

Sudah tersedia:

```text
personalized recommendation
history mode
onboarding mode
FastAPI
health endpoint
validation
automated tests
```

Belum menjadi bagian production flow saat ini:

```text
automatic wishlist weighting
automatic booking weighting
click-event weighting
online retraining
continuous model retraining
real-time preference updater
```

---

# 47. Ringkasan Backend dalam 30 Detik

```text
1. Jalankan/deploy ML FastAPI.
2. Simpan base URL di Laravel.
3. Laravel pilih candidate film.
4. Laravel kirim history/onboarding + metadata.
5. POST /recommendations.
6. ML mengembalikan ranking.
7. Tampilkan berdasarkan rank.
8. Jangan anggap score sebagai persen.
9. Tangani 422, 500, timeout.
10. Laravel tidak perlu menjalankan XGBoost sendiri.
```

---

# 48. Command Cepat

Setup:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Test:

```powershell
python -m pytest -q tests
```

Run API:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Health:

```text
http://127.0.0.1:8001/health
```

Swagger:

```text
http://127.0.0.1:8001/docs
```

Recommendation:

```text
POST http://127.0.0.1:8001/recommendations
```

---

# 49. Arsitektur Akhir

```text
┌───────────────────┐
│     Frontend      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Laravel Backend   │
│                   │
│ auth              │
│ ratings/history   │
│ movie availability│
│ schedules         │
│ price             │
└─────────┬─────────┘
          │ HTTP JSON
          ▼
┌─────────────────────────────┐
│ FastAPI Recommendation ML   │
│                             │
│ PreferenceBuilder           │
│ FeatureBuilder              │
│ XGBRanker                   │
│ MovieRanker                 │
└─────────┬───────────────────┘
          │ ranked JSON
          ▼
┌───────────────────┐
│ Laravel Backend   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     Frontend      │
└───────────────────┘
```

**Backend tidak menggunakan file model secara langsung. Backend menggunakan API yang menjalankan model.**
