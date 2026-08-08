import os
import time
from llama_cpp import Llama

# 1. Tentukan lokasi model GGUF kita
GGUF_MODEL_PATH = os.path.join(os.getcwd(), "models", "Qwen-0.5B-GGUF", "qwen1_5-0_5b-chat-q4_k_m.gguf")


def run_fast_inference():
    print("=== MENGHIDUPKAN PUSTAKAWAN AI (TURBO MODE) ===")

    # Cek apakah file GGUF ada
    if not os.path.exists(GGUF_MODEL_PATH):
        print("[-] File GGUF tidak ditemukan! Jalankan perintah huggingface-cli download terlebih dahulu.")
        return

    print("[*] Memuat model GGUF ke RAM...")
    start_load = time.time()
    # 2. Inisialisasi Llama CPP Engine
    # Ini menggantikan PyTorch dan HuggingFace AutoModel sepenuhnya!
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=2048,  # Context Window: Maksimal token yang bisa diingat (Pertanyaan + Jawaban)
        verbose=False,  # Matikan log C++ yang berisik di terminal
    )
    print(f"[+] Model super ringan termuat dalam {time.time() - start_load:.2f} detik!\n")

    # 3. Menyiapkan Prompt
    pertanyaan = "Tolong jelaskan secara singkat apa itu Kuantisasi dalam AI?"
    # Format prompt standar untuk model Chat (meniru ChatML format)
    prompt = (
        "<|im_start|>system\n"
        "Kamu adalah asisten AI yang cerdas.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{pertanyaan}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print(f"User: {pertanyaan}")
    print("AI : ", end="", flush=True)
    start_infer = time.time()

    # 4. Generate Response (Streaming)
    # Stream=True membuat AI mengetik huruf demi huruf di layar, tidak perlu menunggu sampai selesai berpikir

    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.7,
        stop=["<|im_end|>"],  # Memberitahu engine kapan AI harus berhenti bicara
        stream=True,
    )

    # 5. Menampilkan Streaming Text
    for token in output:
        # Menangkap setiap kata/token yang keluar dari mesin dan mencetaknya ke layar
        teks = token['choices'][0]['text']
        print(teks, end="", flush=True)
    end_infer = time.time()
    print(f"\n\n[*] Waktu inferensi Turbo: {end_infer - start_infer:.2f} detik.")


if __name__ == "__main__":
    run_fast_inference()
