import os
import glob
import time
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
 
# --- KONFIGURASI PATH --- 
RAW_DIR = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "knowledge_master.parquet")
 
# Pastikan model embedding tersimpan di lokal (menghindari C:) 
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), "models",
"sentence_transformers")

def extract_documents():
    """Fase Extract: Membaca semua dokumen dari folder raw."""
    file_paths = glob.glob(os.path.join(RAW_DIR, "*.md")) + glob.glob(os.path.join(RAW_DIR,
"*.txt"))
    documents = []
    for path in file_paths:
        file_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            documents.append({"source": file_name, "text": f.read()})
    print(f"[*] Extract: Menemukan {len(documents)} dokumen.")
    return documents

def transform_chunking(documents):
    """Fase Transform 1: Memotong teks menjadi ukuran kecil yang mudah dicerna AI."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunked_data = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"]) 
    
    for i, chunk in enumerate(chunks):
            chunked_data.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_part_{i}",
                "content": chunk
            })
    print(f"[*] Transform: Menghasilkan {len(chunked_data)} potongan teks (chunks).")
    return chunked_data

def transform_embedding(chunked_data):
    """Fase Transform 2: Mengubah potongan teks menjadi Vektor."""
    print("[*] Transform: Memuat model Embedding ke CPU...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    teks_list = [item["content"] for item in chunked_data]
    print("[*] Transform: Menghitung koordinat Vektor (Mohon tunggu)...")
     
    # Proses komputasi berat di CPU 
    embeddings = model.encode(teks_list, show_progress_bar=True)
     
    # Memasukkan array vektor kembali ke dalam dictionary data 
    for i, item in enumerate(chunked_data):
        item["embedding"] = embeddings[i].tolist()
        
    return chunked_data

def load_to_database(final_data):
    """Fase Load: Menyimpan hasil akhir ke format Parquet berkinerja tinggi."""
    df = pd.DataFrame(final_data)
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
    print(f"[*] Load: Database berhasil diperbarui di -> {OUTPUT_FILE}")

def run_master_pipeline():
    """Fungsi Orkestrasi Utama (Controller)."""
    start_time = time.time()
    print("=== MEMULAI NEURAL DATA PIPELINE ===")
     
    # Rantai Eksekusi Pipeline 
    raw_docs = extract_documents()
    if not raw_docs:
        print("[-] Pipeline berhenti. Tidak ada data.")
        return
        
    chunks = transform_chunking(raw_docs) 

    embedded_data = transform_embedding(chunks)
    load_to_database(embedded_data)

    end_time = time.time()
    print(f"=== PIPELINE SELESAI DALAM {end_time - start_time:.2f} DETIK ===")


if __name__ == "__main__":
    run_master_pipeline()
 