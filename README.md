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

- Untuk Metabase business Dashboard di localhost
  Pastikan Anda memiliki:

  - Docker Desktop
  - Pull metabase image
    ```bash
    docker pull metabase/metabase
    ```
  - Run metabase container
    ```bash
    docker run -p 3030:3000 --name metabase-armada metabase/metabase
    ```
  - Tunggu sampai masuk ke http://localhost:3030/setup dengan sempurna
  - Matikan container
    ```bash
    docker stop metabase-armada
    ```
  - Salin file data configurasi metabase
    ```bash
    docker cp ./metabase.db.mv.db metabase-armada:/metabase.db/
    ```
  - Jalankan kembali container
    ```bash
    docker start metabase-armada
    ```

## Business Dashboard

Dashboard ini menyajikan analisis menyeluruh terhadap performa akademik mahasiswa selama dua semester pertama. Fokus utama dashboard ini adalah mengidentifikasi mahasiswa berisiko tinggi untuk dropout dan memberikan wawasan berbasis data untuk mendukung pengambilan keputusan intervensi akademik dini.

### Cara Akses Dashboard

1. Login ke Metabase

   - Buka browser dan akses: `http://localhost:3030`
   - Masukkan kredensial berikut:

     - Email: `root@mail.com`
     - Password: `root123`

2. Navigasi ke Dashboard

   - Klik menu di sebelah kiri layar
   - Pilih: Collections → Our Analytics
   - Cari dan buka: Student Performance Dashboard

### Fitur Utama Dashboard

1. Ringkasan Performa Akademik

   - Menampilkan rata-rata unit dan nilai yang disetujui pada semester 1 dan 2.
   - Menyediakan angka dropout rate (32.12%) sebagai indikator utama.

2. Korelasi Kinerja Akademik & Status Mahasiswa

   - Visualisasi sebaran mahasiswa berdasarkan nilai dan unit yang disetujui, dikelompokkan berdasarkan status (Dropout, Enrolled, Graduate).

3. Perbandingan Kinerja Akademik Berdasarkan Status

   - Grafik batang yang menunjukkan rata-rata unit dan nilai mahasiswa berdasarkan status akhir mereka.

4. Dropout Rate Berdasarkan Jumlah Unit & Nilai

   - Analisis dropout berdasarkan kelompok unit yang disetujui dan rentang nilai akademik.

5. Penurunan Performa antar Semester

   - Menampilkan tren penurunan unit dan nilai dari semester 1 ke semester 2, terutama pada mahasiswa yang dropout.

6. Analisis Pemulihan Semester 2

   - Visualisasi mahasiswa yang memiliki performa rendah di semester 1 namun menunjukkan pemulihan di semester 2.

7. Dropout Prediction Model (Sankey Diagram)

   - Menunjukkan alur transisi performa mahasiswa dari semester 1 ke semester 2 dan akhirnya ke status akhir.
   - Sangat berguna untuk mengidentifikasi pola yang mengarah ke dropout.

8. Identifikasi Mahasiswa Berisiko Tinggi (Tabel)

   - Tabel interaktif yang menampilkan kombinasi jalur akademik (unit & nilai) dengan jumlah mahasiswa, dropout rate, dan graduation rate.
   - Digunakan untuk menyoroti kelompok mahasiswa yang membutuhkan perhatian dan intervensi segera.

## Menjalankan Sistem Machine Learning

### Menjalankan Secara Lokal

1. Jalankan aplikasi dengan perintah berikut dari terminal (pada folder tempat file `app.py` berada):

   ```bash
   streamlit run app.py
   ```

2. Akses aplikasi di browser melalui alamat:

   ```
   http://localhost:8501
   ```

### Mengakses Secara Online

Prototype ini juga sudah dideploy secara publik menggunakan Streamlit Cloud, sehingga bisa diakses oleh siapa saja tanpa harus menjalankan secara lokal.

- Link Akses Prototype:

[https://und5tyvqgwplsefkfxtk64.streamlit.app/](https://und5tyvqgwplsefkfxtk64.streamlit.app/)

### Deskripsi Antarmuka Aplikasi

#### Sidebar – Model Settings

- Upload model file (.pkl) jika model belum ditemukan di lokal
- Format file: `.pkl`, maksimal 200MB
- Informasi aplikasi:

  - Tujuan: Prediksi hasil akademik mahasiswa
  - Model: `RandomForestClassifier`
  - Output kelas:

    - 0: Dropout
    - 1: Enrolled
    - 2: Graduate

#### Bagian Atas Aplikasi

Judul: 🎓 _Student Dropout Prediction Tool_
Deskripsi: Menjelaskan bahwa aplikasi digunakan untuk memprediksi apakah seorang mahasiswa akan dropout, tetap melanjutkan studi, atau lulus berdasarkan informasi pribadi dan akademik.

Input Data Mahasiswa:

- Demographics
- Application & Course
- Prior Education
- Financial Factors
- 1st Semester Performance
- 2nd Semester Performance

Setiap kategori terdiri dari beberapa field yang harus diisi secara manual oleh pengguna atau gunakan preset:

- Likely Graduate
- At-Risk Student

#### Bagian Bawah Aplikasi – Prediction Results

Setelah data diisi:

- Klik tombol "Generate Prediction"

Hasil yang ditampilkan:

- Prediction Outcome: Dropout / Enrolled / Graduate
- Probability Gauge: Visual pengukur probabilitas
- Probability Chart: Grafik batang atau pie chart
- Deskripsi detail probabilitas (misal: Dropout: 12.0%, Enrolled: 16.0%, Graduate: 72.0%)

#### Faktor yang Mempengaruhi Prediksi

- Disampaikan dalam teks: misalnya mahasiswa yang dropout cenderung memiliki:

  - Jumlah mata kuliah yang lulus rendah, terutama di semester 2
  - Nilai akademik yang rendah atau menurun dari semester 1 ke 2

#### Tabel Input yang Digunakan

Menampilkan semua data input yang telah dimasukkan pengguna dalam bentuk tabel untuk verifikasi.

## Conclusion

Proyek ini berhasil membangun sebuah dashboard analitik performa mahasiswa yang memberikan wawasan mendalam terhadap faktor-faktor yang berkontribusi pada tingginya tingkat dropout (32.12%) di dua semester awal perkuliahan.
Dengan mengintegrasikan metrik akademik (nilai dan unit yang disetujui), status mahasiswa, serta model prediktif visual, dashboard ini dapat secara efektif mengidentifikasi mahasiswa berisiko tinggi untuk dropout sejak dini.

Data menunjukkan bahwa mahasiswa dengan kombinasi unit rendah dan nilai rendah pada semester pertama & kedua memiliki kemungkinan besar untuk mengalami penurunan performa dan akhirnya dropout. Sebaliknya, mahasiswa dengan pemulihan performa di semester kedua memiliki peluang lebih baik untuk bertahan dan lulus.

### Rekomendasi Action Items

Berikut adalah beberapa langkah strategis yang dapat dilakukan oleh institusi pendidikan berdasarkan temuan dalam dashboard ini:

- Implementasikan sistem peringatan dini untuk mahasiswa yang memiliki <3 unit disetujui atau nilai <10 di semester 1.
- Lakukan intervensi akademik terarah seperti remedial, bimbingan belajar, atau konsultasi akademik bagi mahasiswa yang teridentifikasi dalam kategori berisiko tinggi.
- Tingkatkan monitoring dan evaluasi performa semester ke semester, dengan fokus khusus pada mahasiswa yang mengalami penurunan performa.
- Gunakan visualisasi prediktif sebagai alat rapat dosen wali, untuk mendiskusikan perkembangan akademik mahasiswa dan strategi pembimbingan.
- Integrasikan dashboard ini dengan sistem akademik kampus agar dapat digunakan secara otomatis dan real-time oleh pengelola akademik dan dosen pembimbing.
