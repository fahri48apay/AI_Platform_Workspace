import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Definisi Direktori (Mise en Place)
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

# Memastikan folder tersedia (jika belum dibuat lewat terminal)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Inisialisasi Text Splitter (Alat Pemotong)
# Menggunakan parameter chunk_size=150 dan overlap=30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    # Memprioritaskan pemotongan pada paragraf (\n\n), baru kemudian baris baru atau spasi
    separators=["\n\n", "\n", " ", ""] 
)

# List kosong untuk menampung semua potongan teks
all_chunks = []

print("Memulai proses Data Ingestion...")

# 3. PROSES LOAD: Membaca file di folder data/raw/
for filename in os.listdir(RAW_DIR):
    # Kita filter agar hanya membaca file teks atau markdown
    if filename.endswith(".txt") or filename.endswith(".md"):
        file_path = os.path.join(RAW_DIR, filename)
        
        # Buka dan baca isi file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 4. PROSES CHUNK: Memotong dokumen panjang menjadi bagian kecil
            chunks = text_splitter.split_text(content)
            
            # Menyimpan tiap potongan ke dalam list beserta metadata (asal file)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source_file": filename,
                    "chunk_id": i,
                    "text_content": chunk
                })

# 5. PROSES SAVE: Menyimpan hasil ke format Parquet
if all_chunks:
    # Mengubah data dari format list of dictionary menjadi Pandas DataFrame
    df = pd.DataFrame(all_chunks)
    
    # Menentukan lokasi dan nama file output
    output_path = os.path.join(PROCESSED_DIR, "knowledge_base.parquet")
    
    # Menyimpan DataFrame ke format Parquet menggunakan engine pyarrow
    df.to_parquet(output_path, engine="pyarrow")
    

    print(f"Sukses! {len(all_chunks)} potongan (chunks) berhasil disimpan di: {output_path}")
else:
    print("Tidak ada data yang diproses. Pastikan ada file teks di dalam folder /data/raw/")