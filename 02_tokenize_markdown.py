import os
from transformers import AutoTokenizer

#1.Arahkan ke folder LOKAL tempat model Qwen disimpan (bukan internet!)
LOCAL_MODEL_DIR = os.path.join(os.getcwd(),"models","Qwen-0.5B-Chat")
MARKDOWN_FILE = os.path.join(os.getcwd(),"data","knowledge_base.md")

def process_markdown():
    print(f"[*] Memuat Tokenizer lokal dari:{LOCAL_MODEL_DIR}")
    #2.inisialisasi tokenizer dari folder lokal
    #kita menggunakan AutoTokenizer yang akan otomatis membaca file config model
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    #3.Membaca file markdown mentah
    print(f"[*] Membaca dokumen:{MARKDOWN_FILE}")
    with open(MARKDOWN_FILE,"r",encoding="utf-8")as file:
        raw_text = file.read()

    print("\n--- Teks Asli ---")
    print(raw_text)

    #4.Melakukan Tokenisasi (Encoding)
    #Mengubah teks menjadi angka
    encoded_input = tokenizer(raw_text)

    print("\n--- Hasil Tokenisasi (Machine Input) ---")
    print(f"Input IDs (Angka representasi kata):\n{encoded_input['input_ids']}")

    #Menghitung jumlah token
    token_count = len(encoded_input['input_ids'])
    print(f"\nJumlah total token:{token_count}")


    decoded_text = tokenizer.decode(encoded_input['input_ids'])
    print("\n--- Hasil Decoding (Kembali ke Teks) ---")
    print(decoded_text)
    
    
if __name__=="__main__":
    process_markdown()
