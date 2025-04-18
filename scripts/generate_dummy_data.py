import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def generate_dummy_data_func():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Number of samples
    n_samples = 10000

    # Generate data
    age = np.random.randint(18, 70, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    purchase_amount = np.random.normal(1500000, 200000, n_samples)  # mean=1500000, std=200000
    tenure = np.random.randint(1, 120, n_samples)  # in months (1 month to 10 years)

    # Standardize variables to make the effect sizes more controllable
    age_std = (age - np.mean(age)) / np.std(age)
    purchase_amount_std = (purchase_amount - np.mean(purchase_amount)) / np.std(purchase_amount)
    tenure_std = (tenure - np.mean(tenure)) / np.std(tenure)

    # Create gender effect (females have higher churn)
    gender_effect = np.where(gender == 'Female', 1, 0)

    # Model churn probability with strong negative correlations for age, purchase_amount, and tenure
    # Higher coefficients for stronger negative correlations
    churn_prob = (
        - 3 # lower base probability to reduce overall churn
        - 0.5 * age_std  # negative correlation with age
        - 2 * purchase_amount_std  # negative correlation with purchase amount
        - 1 * tenure_std  # stronger negative correlation with tenure
        + gender_effect  # women churn more
        + np.random.normal(0, 0.1, n_samples)  # add some noise
    )

    churn_prob = 1 / (1 + np.exp(-churn_prob))  # sigmoid to keep values between 0 and 1
    churn_prob = np.clip(churn_prob, 0.05, 0.95)  # limit probabilities between 5% and 95%

    # Generate churn label based on probability
    churn = np.random.binomial(1, churn_prob)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'purchase_amount': purchase_amount.round(2),
        'tenure': tenure,
        'churn': churn
    })

    # Export to CSV
    df.to_csv('data/customer_data.csv', index=False)

    print(f"Generated dataset with {n_samples} samples and saved to data/customer_data.csv")
    print("\nData preview:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nChurn distribution:")
    print(df['churn'].value_counts(normalize=True))

    # Check correlation between features and churn
    print("\nCorrelation with churn:")
    numeric_cols = ['age', 'purchase_amount', 'tenure']
    for col in numeric_cols:
        correlation = df[col].corr(df['churn'])
        print(f"{col}: {correlation:.4f}")

    # Encode gender for correlation analysis
    le = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    correlation = df['gender_encoded'].corr(df['churn'])
    print(f"gender: {correlation:.4f}")

    # Check churn rate by gender
    print("\nChurn rate by gender:")
    print(df.groupby('gender')['churn'].mean())
    
    # Return some statistics for display in the app
    return {
        "samples": n_samples,
        "churn_rate": df['churn'].mean() * 100,
        "preview": df.head().to_dict()
    }

if __name__ == "__main__":
    generate_dummy_data_func()