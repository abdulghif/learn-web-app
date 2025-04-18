import os
import sys
import importlib

# Now import the required packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.analysis import run_analysis

# Import the scripts instead of running them as subprocesses
from scripts.generate_dummy_data import generate_dummy_data_func
from scripts.train_model import train_churn_model

# Setting page config
st.set_page_config(
    page_title="Aplikasi Prediksi Churn",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    with open('models/churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi
def predict_churn(age, gender, purchase_amount, tenure):
    # Buat DataFrame dengan format yang sama seperti data training
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'purchase_amount': [purchase_amount],
        'tenure': [tenure]
    })
    
    # Lakukan prediksi
    model = load_model()
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba

# Fungsi untuk menjalankan generate_dummy_data.py
def generate_dummy_data():
    try:
        # Call the function directly instead of as a subprocess
        results = generate_dummy_data_func()
        return True, results
    except Exception as e:
        return False, f"Error: {str(e)}"

# Fungsi untuk menjalankan train_model.py
def train_model():
    try:
        # Call the function directly instead of as a subprocess
        model = train_churn_model()
        return True, "Model successfully trained and saved."
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    # Sidebar untuk navigasi
    st.sidebar.title('Navigasi')
    pages = ["Prediksi Churn", "Analisis Data", "Persiapan Data & Model"]
    selection = st.sidebar.radio("Pilih Halaman:", pages)
    
    if selection == "Prediksi Churn":
        # Main page title
        st.title('Aplikasi Prediksi Churn Pelanggan')
        st.write("""
        Aplikasi ini memprediksi kemungkinan pelanggan akan berhenti (churn) berdasarkan
        beberapa karakteristik pelanggan menggunakan model machine learning.
        """)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Input form in first column
        with col1:
            st.header('Input Data Pelanggan')
            
            age = st.slider('Usia', min_value=18, max_value=80, value=35, step=1)
            gender = st.selectbox('Gender', options=['Male', 'Female'])
            purchase_amount = st.number_input('Jumlah Pembelian (Rp)', min_value=0, max_value=2000000000, value=500000, step=100000)
            tenure = st.slider('Lama Berlangganan (bulan)', min_value=1, max_value=120, value=24, step=1)
            
            predict_button = st.button('Prediksi Churn')
        
        # Results in second column
        with col2:
            if predict_button:
                if os.path.exists('models/churn_model.pkl'):
                    prediction, probability = predict_churn(age, gender, purchase_amount, tenure)
                    
                    st.header('Hasil Prediksi')
                    
                    # Visualisasi gauge untuk probabilitas churn
                    fig, ax = plt.subplots(figsize=(6, 3))
                    
                    # Membuat gauge chart sederhana
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 0.5)
                    ax.set_axis_off()
                    
                    # Tambahkan gauge background
                    ax.barh(0.2, 1, height=0.1, color='lightgrey', alpha=0.5)
                    
                    # Tambahkan gauge bar
                    ax.barh(0.2, probability, height=0.1, color='red' if probability >= 0.5 else 'green')
                    
                    # Tambahkan teks
                    ax.text(0, 0.35, "0%", ha='left', fontsize=12)
                    ax.text(1, 0.35, "100%", ha='right', fontsize=12)
                    ax.text(0.5, 0.35, f"{probability*100:.1f}%", ha='center', fontsize=14, fontweight='bold')
                    
                    # Title
                    ax.text(0.5, 0.45, "Probabilitas Churn", ha='center', fontsize=14)
                    
                    st.pyplot(fig)
                    
                    # Menampilkan hasil keputusan
                    if prediction == 1:
                        st.error("### PELANGGAN BERPOTENSI CHURN")
                        st.write(f"Pelanggan ini memiliki {probability*100:.1f}% kemungkinan akan churn.")
                    else:
                        st.success("### PELANGGAN KEMUNGKINAN TETAP LOYAL")
                        st.write(f"Pelanggan ini memiliki {(1-probability)*100:.1f}% kemungkinan akan tetap loyal.")
                    
                    # Rekomendasi berdasarkan prediksi
                    st.subheader("Rekomendasi:")
                    if prediction == 1:
                        st.write("""
                        1. Hubungi pelanggan untuk diskusi tentang kebutuhan mereka
                        2. Tawarkan diskon atau insentif khusus
                        3. Minta feedback tentang produk/layanan
                        4. Kirimkan penawaran loyalitas khusus
                        """)
                    else:
                        st.write("""
                        1. Pertahankan komunikasi rutin
                        2. Tawarkan program loyalitas
                        3. Tingkatkan pengalaman pelanggan
                        """)
                    
                else:
                    st.error("Model belum tersedia. Silakan kunjungi halaman 'Persiapan Data & Model' untuk membuat data dummy dan melatih model!")
    
    elif selection == "Analisis Data":
        run_analysis()
    
    elif selection == "Persiapan Data & Model":
        st.title('Persiapan Data dan Model')
        st.write("""
        Halaman ini memungkinkan Anda untuk menghasilkan data dummy dan melatih model machine learning.
        """)
        
        # Cek apakah data sudah ada
        data_exists = os.path.exists('data/customer_data.csv')
        model_exists = os.path.exists('models/churn_model.pkl')
        
        st.subheader('Status:')
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Data pelanggan: {'✅ Tersedia' if data_exists else '❌ Belum tersedia'}")
            st.write(f"Model machine learning: {'✅ Tersedia' if model_exists else '❌ Belum tersedia'}")
        
        st.subheader('Langkah 1: Generate Data Dummy')
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gen_data_button = st.button('Generate Data Dummy')
        
        with col2:
            if gen_data_button:
                with st.spinner('Sedang menghasilkan data dummy...'):
                    success, output = generate_dummy_data()
                    if success:
                        st.success("✅ Data dummy berhasil dibuat!")
                        if os.path.exists('data/customer_data.csv'):
                            try:
                                df = pd.read_csv('data/customer_data.csv')
                                st.write(f"Preview data ({df.shape[0]} baris, {df.shape[1]} kolom):")
                                st.dataframe(df.head())
                            except Exception as e:
                                st.error(f"Error membaca data: {e}")
                    else:
                        st.error(f"❌ Gagal menghasilkan data dummy: {output}")
        
        st.subheader('Langkah 2: Latih Model')
        col1, col2 = st.columns([1, 2])
        
        with col1:
            train_button = st.button('Latih Model')
        
        with col2:
            if train_button:
                if not os.path.exists('data/customer_data.csv'):
                    st.error("❌ Data pelanggan belum tersedia. Silakan generate data dummy terlebih dahulu!")
                else:
                    with st.spinner('Sedang melatih model...'):
                        success, output = train_model()
                        if success:
                            st.success("✅ Model berhasil dilatih!")
                            if os.path.exists('models/churn_model.pkl'):
                                st.write("Model tersimpan di 'models/churn_model.pkl'")
                        else:
                            st.error(f"❌ Gagal melatih model: {output}")
        
        if model_exists:
            st.success("✅ Data dan model sudah siap! Silakan kembali ke halaman 'Prediksi Churn' untuk mencoba aplikasi.")

if __name__ == "__main__":
    main()