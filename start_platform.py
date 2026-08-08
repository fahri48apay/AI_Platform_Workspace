import os
import sys
import logging
import subprocess
import uvicorn
# --- KONFIGURASI LOGGING ---
# Menyimpan semua log ke dalam file 'platform.log'
logging.basicConfig(
filename='platform.log',
level=logging.INFO, # Catat mulai dari level INFO ke atas
format='%(asctime)s - %(levelname)s - %(message)s',
datefmt='%Y-%m-%d %H:%M:%S'
)
# Menambahkan log ke layar terminal juga (agar kita tetap bisa melihatnya)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
# --- KONFIGURASI PATH ---
DB_FILE = os.path.join(os.getcwd(), "data", "processed", "knowledge_master.parquet")
MODEL_FILE = os.path.join(os.getcwd(), "models", "Qwen-0.5B-GGUF",
"qwen1_5-0_5b-chat-q4_k_m.gguf")
def orchestrate():
logging.info("="*50)
logging.info("MEMULAI LOCAL AI COMPUTING PLATFORM")
logging.info("="*50)
# 1. Pengecekan Sistem Utama (Model GGUF)
if not os.path.exists(MODEL_FILE):
logging.error(f"FATAL: Model GGUF tidak ditemukan di {MODEL_FILE}")
logging.error("Sistem dihentikan. Silakan jalankan unduhan model terlebih dahulu.")
sys.exit(1)
logging.info("CHECK: Model GGUF tersedia.")
# 2. Pengecekan Database (Auto-Healing)
if not os.path.exists(DB_FILE):
logging.warning("Database Parquet tidak ditemukan! Menginisiasi proses Build ETL...")
try:
# Memanggil skrip Sesi 7 secara otomatis
logging.info("Menjalankan 06_master_pipeline.py...")
subprocess.run(["python", "06_master_pipeline.py"], check=True)
logging.info("Proses ETL selesai. Database berhasil dibuat.")
except subprocess.CalledProcessError:
logging.error("FATAL: Gagal menjalankan proses ETL.")
sys.exit(1)
else:
logging.info("CHECK: Database Parquet tersedia.")
# 3. Menjalankan Server API
logging.info("Menyiapkan peluncuran Uvicorn API Server...")
logging.info("Platform dapat diakses di: http://127.0.0.1:8000/docs")
try:
# Menjalankan aplikasi dari file 11_api_server.py
# Format string: "nama_file_tanpa_py:nama_variabel_FastAPI"
uvicorn.run("11_api_server:app", host="127.0.0.1", port=8000, log_level="info")
except Exception as e:
logging.error(f"Server API mengalami crash: {str(e)}")
if __name__ == "__main__":
orchestrate()