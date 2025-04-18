import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import run_analysis

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

def main():
    # Sidebar untuk navigasi
    st.sidebar.title('Navigasi')
    pages = ["Prediksi Churn", "Analisis Data"]
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
            purchase_amount = st.number_input('Jumlah Pembelian (Rp)', min_value=0, max_value=2000, value=500, step=50)
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
                    st.error("Model belum tersedia. Silakan latih model terlebih dahulu!")
    
    elif selection == "Analisis Data":
        run_analysis()

if __name__ == "__main__":
    main()
