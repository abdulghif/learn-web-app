import streamlit as st
import pandas as pd
import numpy as np

# Menampilkan judul aplikasi
st.title("Aplikasi Streamlit Sederhana")

# Menampilkan teks deskripsi
st.write("Selamat datang! Ini adalah aplikasi sederhana untuk belajar Streamlit.")

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

# Menampilkan grafik sederhana
st.write("Berikut adalah contoh grafik sederhana:")
st.line_chart([1, 2, 3, 4, 5])

# Menampilkan slider untuk memilih angka
slider_value = st.slider("Pilih angka favorit Anda", 0, 100, 50)
st.write(f"Angka favorit Anda adalah: {slider_value}")

# Menampilkan tabel data
st.write("Berikut adalah contoh tabel data:")
data = pd.DataFrame({
    'Kolom A': [1, 2, 3, 4],
    'Kolom B': [10, 20, 30, 40]
})
st.table(data)

# Menampilkan peta dengan titik-titik di Jakarta
st.write("Berikut adalah contoh peta lokasi di Jakarta:")
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

# Menampilkan file uploader
st.write("Unggah file Anda di sini:")
uploaded_file = st.file_uploader("Pilih file")
if uploaded_file is not None:
    st.write("File berhasil diunggah!")