{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# End-to-End Proses Prediksi Churn Pelanggan\n",
    "\n",
    "Notebook ini menunjukkan proses lengkap pembuatan model prediksi churn pelanggan, mulai dari:\n",
    "1. Pembuatan dan analisis data\n",
    "2. Preprocessing data\n",
    "3. Pembuatan dan evaluasi model\n",
    "4. Penyimpanan model untuk aplikasi Streamlit\n",
    "5. Contoh penggunaan model untuk prediksi"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Library yang Diperlukan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score\n",
    "\n",
    "# Set random seed\n",
    "np.random.seed(42)\n",
    "\n",
    "# Konfigurasi visualisasi\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Generate Data Dummy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Buat direktori untuk menyimpan data jika belum ada\n",
    "os.makedirs('../data', exist_ok=True)\n",
    "\n",
    "# Jumlah sampel\n",
    "n_samples = 1000\n",
    "\n",
    "# Generate data\n",
    "age = np.random.randint(18, 70, n_samples)\n",
    "gender = np.random.choice(['Male', 'Female'], n_samples)\n",
    "purchase_amount = np.random.normal(500, 200, n_samples)  # mean=500, std=200\n",
    "tenure = np.random.randint(1, 120, n_samples)  # dalam bulan (1 bulan hingga 10 tahun)\n",
    "\n",
    "# Model probabilitas churn\n",
    "# Formula sederhana: churn tinggi untuk pelanggan baru dengan pembelian rendah\n",
    "churn_prob = 0.8 - (0.005 * tenure) - (0.0005 * purchase_amount) + (0.005 * (age - 40)**2 / 100)\n",
    "churn_prob = np.clip(churn_prob, 0.05, 0.95)  # batasi probabilitas antara 5% dan 95%\n",
    "\n",
    "# Generate label churn berdasarkan probabilitas\n",
    "churn = np.random.binomial(1, churn_prob)\n",
    "\n",
    "# Buat DataFrame\n",
    "df = pd.DataFrame({\n",
    "    'age': age,\n",
    "    'gender': gender,\n",
    "    'purchase_amount': purchase_amount.round(2),\n",
    "    'tenure': tenure,\n",
    "    'churn': churn\n",
    "})\n",
    "\n",
    "# Ekspor ke CSV\n",
    "df.to_csv('../data/customer_data.csv', index=False)\n",
    "\n",
    "print(f\"Generated dataset with {n_samples} samples and saved to data/customer_data.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Eksplorasi Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load data\n",
    "df = pd.read_csv('../data/customer_data.csv')\n",
    "\n",
    "# Informasi dataset\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Statistik deskriptif\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Cek missing values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Distribusi churn\n",
    "plt.figure(figsize=(10, 6))\n",
    "ax = sns.countplot(x='churn', data=df)\n",
    "plt.title('Distribusi Churn')\n",
    "plt.xlabel('Churn (0=Tidak, 1=Ya)')\n",
    "plt.ylabel('Jumlah Pelanggan')\n",
    "\n",
    "# Tambahkan jumlah dan persentase di atas bar\n",
    "total = len(df)\n",
    "for p in ax.patches:\n",
    "    percentage = f'{100 * p.get_height() / total:.1f}%'\n",
    "    x = p.get_x() + p.get_width() / 2\n",
    "    y = p.get_height()\n",
    "    ax.annotate(f'{p.get_height()} ({percentage})', (x, y), ha='center', va='bottom')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Eksplorasi Hubungan antar Fitur"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 1. Age vs Churn\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(x='churn', y='age', data=df)\n",
    "plt.title('Distribusi Usia berdasarkan Status Churn')\n",
    "plt.xlabel('Churn (0=Tidak, 1=Ya)')\n",
    "plt.ylabel('Usia')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 2. Gender vs Churn\n",
    "gender_churn = pd.crosstab(df['gender'], df['churn'], normalize='index') * 100\n",
    "\n",
    "plt.figure(figsize=(10