import pandas as pd
df = pd.read_parquet("data/processed/knowledge_embedded.parquet")# Lihat tipe data dari elemen pertama di kolom embedding
print(type(df['embedding'].iloc[0])) 
# Lihat berapa panjang array-nya (seharusnya 384)
print(len(df['embedding'].iloc[0])) 
