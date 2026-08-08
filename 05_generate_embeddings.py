import os
import pandas as pd
from sentence_transformers import SentenceTransformer

#1.Tentukan direktori Path (Menyambung dari Sesi 5)
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DIR, "knowledge_base.parquet")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "knowledge_embedded.parquet")

#Lokasi simpan model embedding agar tidak masuk ke drive C:
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.getcwd(), "models", "sentence_transformers")

def build_vector_space():
    print("[*]Memulai proses High-Dimensional Embedding...")

    #2. Cek apakah dile dari sesi 5 ada 
    if not os.path.exists(INPUT_FILE):
        print(f"[-] File input tidak ditemukan di: {INPUT_FILE}")

        return
    #3.Membaca data potongan teks 
    df = pd.read_parquet(INPUT_FILE)
    print(f"[+] File berhasil memuat {len(df)} potongan teks dari parquet.")

    #4.Memuat Model Embedding CPU
    #Jika baru pertama kali dijalankan,skrip ini akan mengunduh model ~80MB
    print("[*] Memuat model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    #5Proses konversi (Encoding)
    #Mengambil semua teks dari kolom 'content' dan mengubahnya menjadi vektor
    print("[*] Sedang menerjemahkan teks menjadi vektor matematis (Mohon tunggu)...")
    teks_list = df['text_content'].tolist()

    #Encode mengasilkan matriks/array berukuran (jumlah_dokumen,384)
    embeddings = model.encode(teks_list, show_progress_bar=True)

    #6Menyimpan hasil vektor kembali ke dataframe
    #kita ubah array numpy menjadi list agar bisa disimpan di parquet
    df['embedding'] = embeddings.tolist()
    
    #menampilkan informasi bentuk matriks
    dimensi = len(df['embedding'].iloc[0])
    print(f"[+] Sukses! Setiap teks kini memiliki {dimensi} dimensi")
    
    #Menampilak preview data 
    print('\n--- Preview Data beserta Vektor ---')
    #Hanya menampilkan 5 dimensi pertama agar terminal tidak penuh
    print(df[['source_file','chunk_id','embedding']].head(2).to_string())

    #7. Menyimpan Parquet baru yang sudah punya koordinat
    df.to_parquet(OUTPUT_FILE, engine="pyarrow",index=False)
    print(f"\n[+] Data sudah siap digunakan untuk pencarian! Tersimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_vector_space()