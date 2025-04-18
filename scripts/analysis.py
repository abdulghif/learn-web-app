import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import roc_curve, auc
import os

def run_analysis():
    st.title('Analisis Prediksi Churn Pelanggan')
    
    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('data/customer_data.csv')
    
    df = load_data()
    
    # Sidebar untuk navigasi
    st.sidebar.title('Navigasi')
    pages = ['Eksplorasi Data', 'Analisis Fitur', 'Performa Model']
    page = st.sidebar.radio('Pilih Halaman:', pages)
    
    if page == 'Eksplorasi Data':
        st.header('Eksplorasi Data')
        
        # Tampilkan informasi dasar
        st.subheader('Ikhtisar Data')
        st.write(f"Total Data: {df.shape[0]} baris dan {df.shape[1]} kolom")
        
        # Preview data
        st.subheader('Preview Data')
        st.dataframe(df.head())
        
        # Statistik deskriptif
        st.subheader('Statistik Deskriptif')
        st.dataframe(df.describe())
        
        # Distribusi churn
        st.subheader('Distribusi Churn')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='churn', data=df, ax=ax)
        ax.set_title('Distribusi Churn')
        ax.set_xlabel('Churn (0=Tidak, 1=Ya)')
        ax.set_ylabel('Jumlah Pelanggan')
        
        # Tambahkan angka di atas bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom',
                        xytext = (0, 5), textcoords = 'offset points')
        
        # Hitung persentase
        churn_pct = df['churn'].value_counts(normalize=True) * 100
        st.write(f"Persentase Churn: {churn_pct[1]:.2f}%")
        st.write(f"Persentase Tidak Churn: {churn_pct[0]:.2f}%")
        
        st.pyplot(fig)
        
    elif page == 'Analisis Fitur':
        st.header('Analisis Fitur')
        
        # Analisis hubungan fitur dengan churn
        st.subheader('Hubungan Fitur dengan Churn')
        
        # 1. Age vs Churn
        st.write('### Usia vs Churn')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='churn', y='age', data=df, ax=ax)
        ax.set_title('Distribusi Usia berdasarkan Status Churn')
        ax.set_xlabel('Churn (0=Tidak, 1=Ya)')
        ax.set_ylabel('Usia')
        st.pyplot(fig)
        
        # 2. Gender vs Churn
        st.write('### Gender vs Churn')
        gender_churn = pd.crosstab(df['gender'], df['churn'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        gender_churn.plot(kind='bar', ax=ax)
        ax.set_title('Persentase Churn berdasarkan Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Persentase (%)')
        ax.legend(['Tidak Churn', 'Churn'])
        
        # Tambahkan label persentase di atas bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
            
        st.pyplot(fig)
        
        # 3. Purchase Amount vs Churn
        st.write('### Jumlah Pembelian vs Churn')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='churn', y='purchase_amount', data=df, ax=ax)
        ax.set_title('Distribusi Jumlah Pembelian berdasarkan Status Churn')
        ax.set_xlabel('Churn (0=Tidak, 1=Ya)')
        ax.set_ylabel('Jumlah Pembelian')
        st.pyplot(fig)
        
        # 4. Tenure vs Churn
        st.write('### Lama Berlangganan vs Churn')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='churn', y='tenure', data=df, ax=ax)
        ax.set_title('Distribusi Lama Berlangganan berdasarkan Status Churn')
        ax.set_xlabel('Churn (0=Tidak, 1=Ya)')
        ax.set_ylabel('Lama Berlangganan (bulan)')
        st.pyplot(fig)
        
        # 5. Correlation Matrix
        st.write('### Matriks Korelasi')
        # Create a copy to avoid modifying the original dataframe
        df_corr = df.copy()
        # Convert gender to numeric
        df_corr['gender'] = df_corr['gender'].map({'Male': 0, 'Female': 1})
        
        corr = df_corr.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Matriks Korelasi antar Fitur')
        st.pyplot(fig)
        
    elif page == 'Performa Model':
        st.header('Performa Model')
        
        # Cek apakah model sudah tersedia, jika belum beri instruksi
        model_path = 'models/churn_model.pkl'
        if not os.path.exists(model_path):
            st.warning("Model belum dilatih. Jalankan script train_model.py terlebih dahulu.")
            return
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load data for evaluation
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # ROC Curve
        st.subheader('ROC Curve')
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Feature importance
        st.subheader('Feature Importance')
        if hasattr(model[-1], 'feature_importances_'):
            try:
                # Try to load feature names from saved information
                feature_names_path = 'models/feature_names.pkl'
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'rb') as f:
                        feature_info = pickle.load(f)
                    
                    # Set up feature names list
                    feature_names = []
                    
                    # Add numeric features (no transformation needed for names)
                    feature_names.extend(feature_info['numeric_features'])
                    
                    # Add transformed categorical features
                    for cat_feature in feature_info['categorical_features']:
                        # For gender, we only have one feature after OneHotEncoder with drop='first'
                        # (Male becomes the reference category, only Female gets a column)
                        if cat_feature == 'gender':
                            feature_names.append(f"{cat_feature}_Female")
                        else:
                            # Handle any other categorical features if you add them in the future
                            unique_vals = df[cat_feature].unique()
                            if len(unique_vals) > 1:  # Skip the first category (drop='first')
                                feature_names.extend([f"{cat_feature}_{val}" for val in unique_vals[1:]])
                else:
                    # Fallback: manually define feature names
                    st.warning("Feature names file not found. Using default feature names.")
                    feature_names = ['age', 'purchase_amount', 'tenure', 'gender_Female']
                
                # Get feature importances
                importances = model[-1].feature_importances_
                
                # Ensure lengths match
                if len(importances) != len(feature_names):
                    st.error(f"Mismatch between feature importances ({len(importances)}) and feature names ({len(feature_names)})")
                    # Fallback: use generic names
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                
                # Create DataFrame for visualization 
                feature_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error retrieving feature importances: {str(e)}")
                st.write("For debugging purposes:")
                st.write(f"Model structure: {model}")
                if hasattr(model, 'named_steps'):
                    if 'classifier' in model.named_steps:
                        st.write(f"Classifier feature importances shape: {model.named_steps['classifier'].feature_importances_.shape}")
                    if 'preprocessor' in model.named_steps:
                        st.write(f"Preprocessor transformers: {model.named_steps['preprocessor'].transformers_}")
        else:
            st.write("Feature importance is not available for this model.")

if __name__ == "__main__":
    run_analysis()