import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Impor kelas mesin AI kita dari file Sesi 10 (Pastikan file tersebut ada di folder yang sama)
# Jika kamu menggunakan file Sesi 9, ubah importnya menjadi: from 08_rag_engine import
LocalRAGEngine
try:
from 10_fast_rag_engine import LocalRAGEngine
except ImportError:
print("[-] Harap pastikan kamu memiliki file mesin RAG dari sesi sebelumnya.")
exit(1)
# 1. Inisialisasi Aplikasi Web (Sang Pelayan)
app = FastAPI(
title="Local AI Computing Platform API",
description="Microservice untuk Sistem RAG Internal",
version="1.0.0",
)

# 2. Inisialisasi Koki (Mesin AI)
# Kita load di luar rute agar AI tidak perlu dimuat ulang setiap kali ada pertanyaan masuk
print("[*] Memanaskan Server dan Memuat AI Engine...")
ai_engine = LocalRAGEngine()
# 3. Model Data Validasi (Struktur JSON Pesanan)
class ChatRequest(BaseModel):
query: str
class ChatResponse(BaseModel):
status: str
query: str
answer: str
# 4. Membuat Endpoint (Jalur Komunikasi)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
"""
Endpoint utama untuk bertanya kepada Pustakawan AI.
"""
try:
# Menyerahkan pesanan dari pengguna ke mesin AI
jawaban = ai_engine.chat(request.query)
# Menyajikan kembali hasilnya dalam bentuk JSON
return ChatResponse(
status="success",
query=request.query,
answer=jawaban
)
except Exception as e:
# Jika terjadi kesalahan di dapur (misal RAM penuh)
raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")
# 5. Endpoint Kesehatan Server (Health Check)
@app.get("/api/health")
async def health_check():
return {"status": "ok", "message": "Sistem AI menyala dan siap menerima perintah."}
if __name__ == "__main__":
# Menjalankan server web lokal di port 8000
print("\n=== SERVER API BERJALAN DI: http://localhost:8000 ===")
uvicorn.run(app, host="127.0.0.1", port=8000)