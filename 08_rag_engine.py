import os
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

#Konfigurasi Path
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
DB_FILE = os.path.join(PROCESSED_DIR, "knowledge_master.parquet")
LLM_DIR = os.path.join(os.getcwd(), "models", "Qwen-0.5B-Chat")
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

        # 3. Load LLM & Tokenizer (untuk menjawab pertanyaan)
        print("[*] Memuat model LLM (Qwen-0.5B-Chat)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_DIR, dtype=torch.float32)
        self.llm.to("cpu")
        
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
        """Membangun prompt dan men-generate jawaban dari LLM (dari sesi4)."""

        #4 Prompt Engineering: ini adalah rahasia utama RAG
        system_prompt = ("Kamu adalah asisten perusahaan yang akurat dan sopan."
        "Gunakan HANYA informasi dari [KONTEKS] yang diberikan untuk menjawab pertanyaan."
        "jika jawaban tidak ada didalam [KONTEKS], katakan 'Maaf, informasi tersebut tidak ada didokument perusahaan."
        "jangan pernah mengarang informasi"
        )

        user_prompt = f"[KONTEKS]:\n{context}\n\n[PERTANYAAN]:\n{query}"

        messages = [
            {"role": "system", "content": "Kamu adalah asisten HR yang Ramah."},
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to("cpu")

        print("\n--- Pustakawan AI Sedang Berpikir & Mengetik ---")
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.llm.generate(inputs.input_ids, max_new_tokens=150, #Cukup panjang untuk sebuah jawaban 
            temperature=0.3, #suhu rendah agar AI lebih logis /tidak berkhayal
            pad_token_id=self.tokenizer.pad_token_id
            )

        generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in
range(len(inputs.input_ids))]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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