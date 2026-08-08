import os 
from huggingface_hub import snapshot_download

#1.Tentukan target model generatif (Chatbot Brain)
MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"

#2. Tentukan lokasi folder penyimpanan model didalam Drive D:
LOCAL_DIR = os.path.join(os.getcwd(),"models","Qwen-0.5B-Chat")

def download_chat_model():
    print(f"[*] Menyiapkan ruang untuk Pustakawan AI kita:{MODEL_ID}")
    print(f"[*] Destinasi Folder:{LOCAL_DIR}")

    try:
        #3.Proses pengunduan selektif
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR, 
            local_dir_use_symlinks=False,
            #Keamanan ekstra:Kita HANYA mengunduh file safetensors dan config
            #Mengabaikan file bobot lama (.bin, .h5) yang beresiko
            ignore_patterns=["*.msgpack","*.bin", "*.h5"]
        )
        print("\n[+]Berhasil! Otak Chatbot telah tersimpan secara lokal di:")
        print(model_path)

    except Exception as e:
        print(f"\n[-]Gagal mengunduh model.Error:{e}")


if __name__=="__main__" :
    download_chat_model()

