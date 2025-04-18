# Aplikasi Prediksi Churn Pelanggan

Aplikasi ini merupakan sistem prediksi churn pelanggan dengan menggunakan machine learning dan diimplementasikan melalui antarmuka web Streamlit.

## Struktur Direktori

```
churn_prediction_app/
│
├── data/
│   └── customer_data.csv      # Data pelanggan untuk modeling
│
├── models/
│   ├── churn_model.pkl        # Model yang telah dilatih 
│   └── feature_names.pkl      # Informasi fitur untuk inference
│
├── scripts/
│   ├── train_model.py         # Script pelatihan model
│   ├── app.py                 # Aplikasi Streamlit utama
│   ├── analysis.py            # Script analisis data
│   └── helper.py              # Fungsi-fungsi pembantu
│
├── notebooks/
│   └── end_to_end_process.ipynb  # Proses lengkap dalam notebook
│
└── README.md                  # Dokumentasi proyek
```

## Fitur Utama

1. **Prediksi Churn**: Memprediksi kemungkinan pelanggan akan berhenti (churn) berdasarkan karakteristik pelanggan:
   - Usia
   - Jenis Kelamin
   - Jumlah Pembelian
   - Lama Berlangganan

2. **Analisis Data**: Menampilkan visualisasi dan analisis data untuk memahami pola churn pelanggan.

3. **Rekomendasi**: Memberikan rekomendasi strategi retensi pelanggan berdasarkan hasil prediksi.

## Petunjuk Penggunaan

### 1. Instalasi Lingkungan

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install library yang diperlukan
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 2. Persiapan Data dan Model

```bash
# Generate data dummy
python scripts/generate_dummy_data.py

# Latih model
python scripts/train_model.py
```

### 3. Menjalankan Aplikasi

```bash
# Jalankan aplikasi streamlit
streamlit run scripts/app.py
```

### 4. Menggunakan Notebook

Anda juga dapat menjelajahi proses end-to-end dengan notebook Jupyter:

```bash
jupyter notebook notebooks/end_to_end_process.ipynb
```

## Fitur-fitur yang Digunakan

Model prediksi churn menggunakan fitur-fitur berikut:

1. **Age (Usia)**: Usia pelanggan dalam tahun
2. **Gender (Jenis Kelamin)**: Jenis kelamin pelanggan (Male/Female)
3. **Purchase Amount (Jumlah Pembelian)**: Total nilai pembelian pelanggan
4. **Tenure (Lama Berlangganan)**: Lama pelanggan telah berlangganan dalam bulan

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Pandas & NumPy**: Manipulasi dan analisis data
- **Scikit-learn**: Library machine learning untuk membuat model prediksi
- **Matplotlib & Seaborn**: Visualisasi data
- **Streamlit**: Framework untuk membuat aplikasi web interaktif
- **Jupyter Notebook**: Untuk analisis eksploratori dan dokumentasi proses end-to-end

## Pengembangan Lanjutan

Beberapa ide untuk pengembangan aplikasi selanjutnya:

1. Integrasi dengan database eksternal untuk data pelanggan real-time
2. Penambahan fitur-fitur lain seperti data transaksi dan interaksi pelanggan
3. Implementasi model yang lebih kompleks seperti XGBoost atau Deep Learning
4. Penambahan fitur export hasil prediksi ke CSV atau PDF
5. Pembuatan dashboard yang lebih interaktif dengan drill-down analysis

## Kontributor

- [Nama Anda] - Pengembang Utama

## Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail.
