import os 
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

#Konfigurasi Path
PROCESSED_DIR = os.path.join(os.getcwd(),"data", "processed")
DB_FILE = os.path.join(PROCESSED_DIR, "knowledge_master.parquet")

os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "models", "sentence_transformers")

def cosine_similarity(query_vec,db_vectors):
    """
    Menghitung kemiripan susdut antara 1 vektor pertanyaan dengan SEMUA vektor dokumen di database.
    Perhitungan menggunakan NumPy matrix multiplication agar super cepat (C-backend).
    """
    # Menghitung dot product (A.B)
    dot_product = np.dot(db_vectors, query_vec)

    #Menghitung panjang/norma dar masing-masing vektor (||A|| dan ||B||)
    query_norm = np.linalg.norm(query_vec)
    db_norms = np.linalg.norm(db_vectors, axis=1)

    # Cosine similarity = Dot Product / (Norm A * Norm B)
    similarities = dot_product / (db_norms * query_norm)
    return similarities

def run_search_engine(user_query, top_k=2):
    print(f"\n[*] User Bertanya: '{user_query}'")

    #1.Memuat database vektor
    if not os.path.exists(DB_FILE):
        print("[-] Database Parquet tidak ditemukan! jalankan sesi 7 terlebih dahulu.")
        return
    df=pd.read_parquet(DB_FILE)

    #Mengubah kolom embbedding (Yang berbentuk list) menjadi matriks NumPy (2D array)
    db_vectors = np.stack(df['embedding'].values)

    #2.Membuat model dan mengubah Pertanyaan (query) menjadi vektor
    print("[*] Memuat model Embedding ke CPU...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("[*] Menerjemahkan pertanyaan ke ruang vektor...")
    query_vector = model.encode(user_query)
    
    # 3. Menghitung Cosine Similarity
    print("[*] Mencari di database (Scanning...)")
    similarities = cosine_similarity(query_vector, db_vectors)
    # Menyimpan skor ke dalam tabel DataFrame
    df['similarity_score'] = similarities
    
    # 4. Mengurutkan hasil dari yang paling mirip (Skor tertinggi)
    # top_k berarti kita hanya mengambil sejumlah K dokumen teratas
    top_results = df.sort_values(by='similarity_score', ascending=False).head(top_k)
    
    # 5. Menampilkan Hasil
    print("\n=== HASIL PENCARIAN TERATAS ===")
    for index, row in top_results.iterrows():
        print(f"Skor Kemiripan: {row['similarity_score']:.4f}")
        print(f"Sumber Dokumen: {row['source']} (ID: {row['chunk_id']})")
        print(f"Isi Teks : {row['content']}")
        print("-" * 40)
    
if __name__ == "__main__":
    # Kamu bisa mengganti kalimat di bawah ini untuk bereksperimen
    pertanyaan = "Gimana prosedur kalau saya mau kerja dari rumah?"
    run_search_engine(user_query=pertanyaan, top_k=2)