# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis

Tingginya angka dropout pada institusi pendidikan (melebihi 30%)

### Cakupan Proyek

- Penggunaan Dataset: Memanfaatkan data yang telah disediakan oleh institusi pendidikan Jaya Jaya untuk analisis lebih lanjut.
- Identifikasi Faktor-Faktor Penyebab: Menganalisis data yang tersedia untuk menemukan variabel yang berkontribusi terhadap tingginya dropout.
- Pengembangan Business Dashboard: Membuat alat visualisasi yang membantu departemen HR dalam memantau faktor-faktor yang mempengaruhi attrition rate.
- Pengembangan Satu Solusi Machine Learning yang Siap Digunakan menggunakan Streamlit UI

### Persiapan

Sumber data:  
https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv

Setup environment:

- Untuk Notebook (`notebook.ipynb`)  
  Jalankan seluruh sel menggunakan Google Colab.  
  Tidak memerlukan instalasi tambahan karena dependensi tersedia di runtime Colab.

- Untuk Prediksi Machine Learning (`app.py`) menggunakan Streamlit di localhost
  Pastikan Anda memiliki:

  - Python 3.11.x
  - Buat Environment Baru:
    ```bash
    python -m venv myenv
    myenv\Scripts\activate.bat
    ```
  - Library yang dibutuhkan:
    ```bash
    pip install requirements.txt
    ```
  - File yang diperlukan:

    - `app.csv`: Data untuk diprediksi.
    - `model/best_model.pkl`: Model machine learning yang telah dilatih.

  - Jalankan skrip dengan:
    ```bash
    streamlit run app.py
    ```

- Untuk Metabase business Dashboard di localhost
  Pastikan Anda memiliki:

  - Docker Desktop
  - Pull metabase image
    ```bash
    docker pull metabase/metabase
    ```
  - Run metabase container
    ```bash
    docker run -p 3030:3000 --name metabase_armada metabase/metabase
    ```
  - Salin file data configurasi metabase
    ```bash
    docker cp ./metabase.db.mv.db metabase_armada:/metabase.db/
    ```

## Business Dashboard

Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Menjalankan Sistem Machine Learning

Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```

```

## Conclusion

Jelaskan konklusi dari proyek yang dikerjakan.

### Rekomendasi Action Items

Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.

- action item 1
- action item 2
