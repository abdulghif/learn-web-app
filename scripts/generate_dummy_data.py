import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set random seed untuk reproduksibilitas
np.random.seed(42)

# Jumlah sampel
n_samples = 1000

# Generate data
age = np.random.randint(18, 70, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
purchase_amount = np.random.normal(500, 200, n_samples)  # mean=500, std=200
tenure = np.random.randint(1, 120, n_samples)  # dalam bulan (1 bulan hingga 10 tahun)

# Model probabilitas churn
# Formula sederhana: churn tinggi untuk pelanggan baru dengan pembelian rendah
churn_prob = 0.8 - (0.005 * tenure) - (0.0005 * purchase_amount) + (0.005 * (age - 40)**2 / 100)
churn_prob = np.clip(churn_prob, 0.05, 0.95)  # batasi probabilitas antara 5% dan 95%

# Generate label churn berdasarkan probabilitas
churn = np.random.binomial(1, churn_prob)

# Buat DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'purchase_amount': purchase_amount.round(2),
    'tenure': tenure,
    'churn': churn
})

# Ekspor ke CSV
df.to_csv('data/customer_data.csv', index=False)

print(f"Generated dataset with {n_samples} samples and saved to data/customer_data.csv")
print("\nData preview:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nChurn distribution:")
print(df['churn'].value_counts(normalize=True))

# Periksa korelasi antara fitur dan churn
print("\nCorrelation with churn:")
numeric_cols = ['age', 'purchase_amount', 'tenure']
for col in numeric_cols:
    correlation = df[col].corr(df['churn'])
    print(f"{col}: {correlation:.4f}")

# Enkode gender untuk analisis korelasi
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])
correlation = df['gender_encoded'].corr(df['churn'])
print(f"gender: {correlation:.4f}")
