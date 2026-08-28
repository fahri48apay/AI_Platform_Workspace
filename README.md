# AI_Platform_Workspace — Local AI Computing Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/RAG-Local--first-4E7CFF?style=for-the-badge)
![Git LFS](https://img.shields.io/badge/Git%20LFS-F05133?style=for-the-badge&logo=gitlfs&logoColor=white)

> **Platform RAG (Retrieval-Augmented Generation) 100% lokal** — dari unduhan model,
> tokenisasi, chunking, embedding, vector search, sampai API server. Semua berjalan di
> **CPU** tanpa cloud dan tanpa kunci API.
>
> Proyek Penilaian Tengah Semester (PTS) — mata kuliah **AI Computing Platform**.

---

## 🧠 Apa yang dibangun?

Sebuah **pustakawan AI pribadi**: dokumen yang kita punya (`data/knowledge_base.md`)
dipotong, diubah jadi vektor, disimpan sebagai database `parquet`, lalu *ditanyai*
menggunakan model bahasa **Qwen 1.5-0.5B** yang berjalan penuh di mesin lokal.

```
data/knowledge_base.md                        models/Qwen-0.5B-Chat
        │ 01 fetch model ◄──────────────────────────────┘
        │ 02 tokenize + 04 chunking (150/30)
        ▼
data/processed/knowledge_base.parquet
        │ 05 embedding (384 dimensi)
        ▼
data/processed/knowledge_embedded.parquet
        │ 06 master ETL (extract → transform → load)
        ▼
data/processed/knowledge_master.parquet
        │ 07 vector search (cosine similarity)
        ▼
  08 RAG engine (CPU) ──► 10 Fast RAG (GGUF q4_K_M) ──► 11 FastAPI
        │                                                     ▲
        └──── start_platform.py (orkestrator + auto-heal) ────┘
                           API di http://127.0.0.1:8000/docs
```

**Alur satu perintah:** `python start_platform.py` → cek model GGUF → *auto-heal*
database (jalankan ETL otomatis bila `parquet` hilang) → luncurkan API server.

---

## ✨ Fitur

| Fitur | Keterangan |
| --- | --- |
| 🏠 **100% lokal & privat** | Model, database, dan API berjalan di mesin sendiri — tidak ada data keluar |
| ⚡ **Tanpa GPU** | Inferensi CPU via PyTorch & `llama-cpp` (GGUF `q4_K_M` — 4-bit quantized) |
| 📦 **Tanpa kunci API** | Bebas biaya langganan LLM cloud |
| 🔧 **Pipeline modular** | 11 skrip berurutan `01` → `11`, tiap langkah dapat dijalankan & diuji mandiri |
| 🧩 **RAG end-to-end** | Chunking → embedding 384-d → cosine similarity → generation |
| 🛡️ **Orkestrator auto-heal** | `start_platform.py` memperbaiki database yang hilang secara otomatis |
| 📝 **Logging terpusat** | Semua aktivitas tercatat di `platform.log` |
| 🌐 **API presentable** | Dokumentasi interaktif otomatis di `/docs` (Swagger UI) |

---

## 📁 Struktur Proyek

```
AI_Platform_Workspace/
├── 01_fetch_llm.py            # Unduh model Qwen1.5-0.5B-Chat dari HuggingFace
├── 02_tokenize_markdown.py    # Tokenisasi dokumen dengan AutoTokenizer lokal
├── 03_cpu_inference.py        # Eksperimen inferensi model di CPU
├── 04_ingest_data.py          # Chunking teks (chunk 150, overlap 30) → parquet
├── 05_generate_embeddings.py  # Buat vektor 384 dimensi (sentence-transformers)
├── 06_master_pipeline.py      # ETL master: extract → transform → load final
├── 07_vector_search.py        # Cosine similarity (perkalian matriks NumPy)
├── 08_rag_engine.py           # Mesin RAG lokal (parquet + Qwen PyTorch)
├── 09_quantized_inference.py  # Inferensi cepat model GGUF via llama-cpp
├── 10_fast_rag_engine.py      # Mesin RAG "turbo" (GGUF 4-bit + parquet)
├── 11_api_server.py           # FastAPI: Local AI Computing Platform API v1.0.0
├── start_platform.py          # Orkestrator: cek model → auto-heal DB → API
├── Cek_tipe_data_parquet.py   # Util: verifikasi panjang embedding (384)
├── data/
│   ├── knowledge_base.md      # Dokumen sumber pengetahuan
│   └── processed/             # Output pipeline: *_knowledge.parquet
└── models/
    ├── Qwen-0.5B-Chat/        # Model asli (safetensors) untuk PyTorch
    ├── Qwen-0.5B-GGUF/        # Model kuantisasi q4_k_m (unduh terpisah — lihat Quick Start)
    └── sentence_transformers/ # Cache embedding (di-ignore git)
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/fahri48apay/AI_Platform_Workspace.git
cd AI_Platform_Workspace

# 1) Environment (disarankan)
python -m venv .venv && source .venv/bin/activate

# 2) Dependensi — PyTorch CPU dulu supaya ringan
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub sentence-transformers langchain-text-splitters \
            llama-cpp-python pandas numpy pyarrow "fastapi[standard]" uvicorn

# 3) Jalankan pipeline langkah demi langkah
python 01_fetch_llm.py          # sekali saja: unduh model ke models/
python 02_tokenize_markdown.py  # tokenisasi dokumen
python 04_ingest_data.py        # chunking
python 05_generate_embeddings.py
python 06_master_pipeline.py    # bangun database vektor final
python 07_vector_search.py      # uji pencarian kosinus

# 4) Sekali saja: unduh model kuantisasi GGUF ke models/
huggingface-cli download Qwen/Qwen1.5-0.5B-Chat-GGUF qwen1_5-0_5b-chat-q4_k_m.gguf \
    --local-dir models/Qwen-0.5B-GGUF

# 5) Atau langsung jalankan seluruh platform (auto-heal DB + API)
python start_platform.py
# → buka http://127.0.0.1:8000/docs
```

> **Catatan model:** `data/processed/*.parquet` sudah disertakan di repo melalui
> **Git LFS** (pastikan `git lfs install` aktif saat clone). Model kuantisasi
> `qwen1_5-0_5b-chat-q4_k_m.gguf` **tidak lagi dibundel** — unduh dari
> Hugging Face dan simpan ke `models/Qwen-0.5B-GGUF/` sebelum menjalankan
> `start_platform.py`.

---

## 🗺️ Peta Pipeline (per skrip)

| # | Skrip | Tugas | Output |
| --- | --- | --- | --- |
| 01 | `fetch_llm` | Unduh `Qwen/Qwen1.5-0.5B-Chat` (snapshot) | `models/Qwen-0.5B-Chat/` |
| 02 | `tokenize_markdown` | Tokenisasi dokumen lokal | statistik token |
| 03 | `cpu_inference` | Uji langsung model di CPU | respons model |
| 04 | `ingest_data` | Potong teks 150 char, overlap 30 | `knowledge_base.parquet` |
| 05 | `generate_embeddings` | Embedding 384-d | `knowledge_embedded.parquet` |
| 06 | `master_pipeline` | ETL konsolidasi sesi 1–6 | `knowledge_master.parquet` |
| 07 | `vector_search` | Cosine similarity seluruh DB | skor kemiripan |
| 08 | `rag_engine` | Mesin RAG lokal (PyTorch) | jawaban ber-sumber |
| 09 | `quantized_inference` | Inferensi `llama-cpp` (q4_K_M) | respons cepat |
| 10 | `fast_rag_engine` | RAG turbo (GGUF) | jawaban ber-sumber |
| 11 | `api_server` | FastAPI microservice | API `:8000/docs` |

---

## 🛠️ Teknologi Utama

| Teknologi | Peran |
| --- | --- |
| **Transformers + PyTorch (CPU)** | Muat & jalankan Qwen 1.5-0.5B |
| **llama-cpp-python** | Inferensi model GGUF kuantisasi 4-bit |
| **sentence-transformers** | Produksi embedding 384 dimensi |
| **langchain-text-splitters** | `RecursiveCharacterTextSplitter` (150/30) |
| **NumPy** | Cosine similarity via matriks (backend C) |
| **pandas + pyarrow** | Database vektor `parquet` |
| **FastAPI + Uvicorn** | API microservice `localhost:8000` |
| **HuggingFace Hub** | Distribusi model (`snapshot_download`) |
| **Git LFS** | Versioning file model & database |

---

## 📌 Informasi Akademik

| | |
| --- | --- |
| **Nama** | Mohammad Fahri Saleh |
| **NIM** | 241101100019 |
| **Kelas** | Sabtu |
| **Semester** | IV |
| **Jurusan** | Teknik Informatika |
| **Mata kuliah** | AI Computing Platform (Proyek PTS) |

---

## 📄 Lisensi

© 2025 **Mohammad Fahri Saleh** — Proyek PTS AI Computing Platform.
Model `Qwen1.5-0.5B-Chat` dilisensikan oleh Alibaba Group sesuai lisensi asli di
`models/Qwen-0.5B-Chat/LICENSE`.