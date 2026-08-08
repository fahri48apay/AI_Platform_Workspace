import os
import time
import numpy as np
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

#Konfigurasi Path
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
DB_FILE = os.path.join(PROCESSED_DIR, "knowledge_master.parquet")
GGUF_MODEL_PATH = os.path.join(os.getcwd(), "models", "Qwen-0.5B-GGUF", "qwen1_5-0_5b-chat-q4_k_m.gguf")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "models", "sentence_transformers")

class LocalRAGEngine:
    def __init__(self):
        print("=== INISIALISASI MESIN RAG LOKAL ===")

        # 1.Load Database
        print("[*] Memuat database vektor...")
        self.df = pd.read_parquet(DB_FILE)
        self.db_vectors = np.stack(self.df['embedding'].values)

        # 2. Load Embedding Model (untuk pencarian)
        print("[*] Memuat model pencarian (MiniLm)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # 3. Load LLM GGUF (untuk menjawab pertanyaan)
        print("[*] Memuat model LLM GGUF (Qwen-0.5B-GGUF)...")
        if not os.path.exists(GGUF_MODEL_PATH):
            raise FileNotFoundError(f"Model GGUF tidak ditemukan: {GGUF_MODEL_PATH}")

        self.llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=2048,
            verbose=False,
        )

        print("[+] Semua Sistem Siap Digunakan!\n")

    def retrieve_context(self, query, top_k=2):
        """Mencari dokumen paling relevan (dari sesi 8)."""
        query_vec = self.embedder.encode(query)
        dot_products = np.dot(self.db_vectors, query_vec)
        norms = np.linalg.norm(self.db_vectors, axis=1) * np.linalg.norm(query_vec)
        similarities = dot_products / norms

        self.df['score'] = similarities
        top_docs = self.df.sort_values(by='score', ascending=False).head(top_k)
        context_text = "\n\n".join(top_docs['content'].tolist())
        return context_text

    def generate_answer(self, query, context):
        """Membangun prompt dan men-generate jawaban dari LLM GGUF."""

        #4 Prompt Engineering: ini adalah rahasia utama RAG
        prompt = (
            "<|im_start|>system\n"
            "Kamu adalah asisten perusahaan yang akurat dan sopan. "
            "Gunakan HANYA informasi dari [KONTEKS] yang diberikan untuk menjawab pertanyaan. "
            "Jika jawaban tidak ada di dalam [KONTEKS], katakan 'Maaf, informasi tersebut tidak ada di dokument perusahaan.' "
            "Jangan pernah mengarang informasi.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"[KONTEKS]:\n{context}\n\n[PERTANYAAN]:\n{query}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        print("\n--- Pustakawan AI Sedang Berpikir & Mengetik ---")
        start_time = time.time()

        output = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.3,
            stop=["<|im_end|>"],
        )

        response = output["choices"][0]["text"].strip()
        print(f"[*] Waktu inferensi: {time.time() - start_time:.2f} detik.")
        return response
    def chat(self, user_query):
        """Orkestrasi: User Bertanya -> Cari Konteks -> Jawab"""
        print(f"\nUser : {user_query}")
        context = self.retrieve_context(user_query)
        print(f"[*] Menemukan {len(context.split())} kata relevan dari database.")
        answer = self.generate_answer(user_query, context)
        print(f"\nAI : {answer}")
        print("-" * 50)

if __name__ == "__main__":
        # Inisialisasi engine (Pemuatan model membutuhkan waktu beberapa detik)
        rag = LocalRAGEngine()

        # Uji Coba Pertanyaan
        rag.chat("Berapa hari jatah cuti saya?")
        rag.chat("Apakah saya boleh WFH setiap hari?")
        rag.chat("Berapa jarak dari bumi ke bulan?") # Pertanyaan jebakan (Out of Context)