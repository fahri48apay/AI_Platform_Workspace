import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "models", "Qwen-0.5B-Chat")

def run_cpu_inference():
    print("[*] Membangun Pustakawan AI (Loading Model Ke RAM)...")
    
    # Validasi path model
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"[ERROR] Model tidak ditemukan di: {LOCAL_MODEL_DIR}")
        print(f"[INFO] Path saat ini: {os.getcwd()}")
        print(f"[INFO] Folder tersedia: {os.listdir('.')}")
        return
    
    start_load = time.time()

    #1.Inisialisasi Tokenizer dan Model
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    
    # Set pad token jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #2.Inisialisasi Model ke CPU
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR, dtype=torch.float32)
    model = model.to("cpu")

    print(f"[+] Model Berhasil dimuat dalam {time.time() - start_load:.2f} detik.\n")

    #3.Menyiapkan Prompt (Pertanyaan)
    #karena ini model chat, kita harus menggunakan format pesan

    messages = [
        {"role": "system", "content": "Kamu adalah asisten yang membantu menjawab pertanyaan."},
        {"role": "user", "content": " jatah cuti karyawan?"}
    ]

    #mengubah format pesan menjadi string yang dipahami model Qwen, lalu di-tokenisasi
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to("cpu") #Memastikan input juga di CPU

    print("--- Pustakawan Sedang Mengetik ---")
    start_infer=time.time()

    #4.Proses inferensi (generate response)
    #torch.no_grad() sangat penting untuk menghemat RAM
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # Tambahkan attention mask
            max_new_tokens=256,# maksimal kata yang dihasilkan
            temperature=0.5, # Tingkat kreativitas (0.1 kaku, 0.3 seimbang, 0.9 kreatif)
            top_p=0.9,  # Nucleus sampling untuk hasil lebih konsisten
            pad_token_id=tokenizer.eos_token_id
        )
    #5.Memisahkan pertanyaan dari jawaban, lalu dekode kembali ke teks
    generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    end_infer=time.time()
    print(f"{response}\n")
    print(f"[*] Waktu inferensi: {end_infer - start_infer:.2f} detik.")



if __name__ == "__main__":
    run_cpu_inference()