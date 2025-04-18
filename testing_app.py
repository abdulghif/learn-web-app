import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Menampilkan judul aplikasi
st.title("Aplikasi Streamlit Sederhana")

# Menampilkan teks deskripsi
st.write("Selamat datang! Ini adalah aplikasi sederhana untuk belajar Streamlit.")

# ---- Bagian Profil Pengguna ----
st.header("📝 Profil Pengguna")

# Input teks: pengguna dapat memasukkan nama mereka
name = st.text_input("Masukkan nama Anda", "Contoh: Budi")

# Input angka: pengguna dapat memasukkan usia mereka
age = st.number_input("Masukkan usia Anda", min_value=0, max_value=120, value=25)

# Checkbox: pengguna dapat memilih apakah mereka menyukai Streamlit
like_streamlit = st.checkbox("Apakah Anda suka Streamlit?")

# Tombol: jika ditekan, akan menampilkan hasil
if st.button("Kirim"):
    # Menampilkan pesan berdasarkan input pengguna
    st.write(f"Halo {name}, usia Anda {age} tahun.")
    if like_streamlit:
        st.write("Senang mendengar Anda menyukai Streamlit!")
    else:
        st.write("Semoga Anda akan menyukai Streamlit setelah mencoba lebih banyak fitur!")

# ---- Bagian Grafik ----
st.header("📊 Visualisasi Data")

# Menampilkan grafik sederhana
st.subheader("Contoh Grafik Sederhana")
st.line_chart([1, 2, 3, 4, 5])

# Menampilkan slider untuk memilih angka
st.subheader("Input Slider")
slider_value = st.slider("Pilih angka favorit Anda", 0, 100, 50)
st.write(f"Angka favorit Anda adalah: {slider_value}")

# ---- Bagian Tabel Data ----
st.header("📋 Tabel Data")

# Menampilkan tabel data
st.subheader("Contoh Tabel Data")
data = pd.DataFrame({
    'Kolom A': [1, 2, 3, 4],
    'Kolom B': [10, 20, 30, 40]
})
st.table(data)

# ---- Bagian Peta ----
st.header("🗺️ Peta Lokasi")

# Pilihan untuk menggunakan data peta default atau input manual
peta_option = st.radio(
    "Pilih mode input untuk peta:",
    ("Data default", "Input manual")
)

if peta_option == "Data default":
    # Menampilkan peta dengan titik-titik di Jakarta
    st.subheader("Peta Lokasi di Jakarta (Data Default)")
    map_data = pd.DataFrame({
        'lat': [-6.2088, -6.1751, -6.2297, -6.1954, -6.3292],
        'lon': [106.8456, 106.8650, 106.8098, 106.8271, 106.8244],
        'location': ['Monas', 'Grand Indonesia', 'Kota Tua', 'Thamrin', 'Ragunan']
    })

    # Tampilkan data lokasi
    st.write("Lokasi yang ditampilkan di peta:")
    st.dataframe(map_data[['location', 'lat', 'lon']])

    # Tampilkan peta
    st.map(map_data)
else:
    # Input manual untuk latitude dan longitude
    st.subheader("Input Lokasi Manual")
    
    col1, col2 = st.columns(2)
    with col1:
        manual_lat = st.number_input("Latitude", value=-6.2088, format="%.4f")
    with col2:
        manual_lon = st.number_input("Longitude", value=106.8456, format="%.4f")
    
    manual_location = st.text_input("Nama Lokasi", "Lokasi Saya")
    
    # Buat dataframe untuk peta
    manual_map_data = pd.DataFrame({
        'lat': [manual_lat],
        'lon': [manual_lon],
        'location': [manual_location]
    })
    
    # Tampilkan data lokasi
    st.write("Lokasi yang akan ditampilkan di peta:")
    st.dataframe(manual_map_data[['location', 'lat', 'lon']])
    
    # Tampilkan peta
    st.map(manual_map_data)

# ---- Bagian Unggah File ----
st.header("📁 Unggah File")

# Periksa dan buat folder data jika belum ada
data_folder = Path("data")
if not data_folder.exists():
    data_folder.mkdir(parents=True)
    st.info("Folder 'data' telah dibuat untuk menyimpan file unggahan.")

# Menampilkan file uploader
uploaded_file = st.file_uploader("Pilih file untuk diunggah")

if uploaded_file is not None:
    # Dapatkan nama file
    file_name = uploaded_file.name
    file_path = os.path.join("data", file_name)
    
    # Simpan file ke folder data
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File '{file_name}' berhasil diunggah dan disimpan ke folder 'data'!")
    
    # Tampilkan informasi file
    file_stats = os.stat(file_path)
    st.write(f"Ukuran file: {file_stats.st_size / 1024:.2f} KB")
    
    # Jika file adalah CSV, tampilkan preview
    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            st.subheader("Preview Data CSV:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Tidak dapat membaca file CSV: {e}")