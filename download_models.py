import os
import gdown

models = {
    "bert_hoax_classifier.pth":
    "https://drive.google.com/file/d/1VvJ5eeBVmVQRV554u2dIaXo1qUnILRdB/view?usp=sharing",
    

    "svm_model.pkl":
    "https://drive.google.com/file/d/1ATjsuRnq6_JaKJFZHY5NORq6FXyif_WG/view?usp=sharing",


    "tfidf_vectorizer.pkl":
    "https://drive.google.com/file/d/1HQoNhTTxGgk9-XewbeZ0DwAWfjA5PRya/view?usp=sharing"
}

for filename, url in models.items():

    temp_file = filename + ".tmp"

    print(f"Mengunduh model terbaru: {filename}")

    downloaded = gdown.download(
        url=url,
        output=temp_file,
        quiet=False
    )

    if (
        downloaded is None
        or not os.path.exists(temp_file)
        or os.path.getsize(temp_file) == 0
    ):
        raise RuntimeError(
            f"Gagal mengunduh {filename}. Periksa tautan Google Drive."
        )

    # Mengganti file lama dengan file terbaru
    os.replace(temp_file, filename)

    print(f"{filename} berhasil diperbarui.")