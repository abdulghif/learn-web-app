import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_churn_model():
    # Buat direktori models jika belum ada
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/customer_data.csv')
    
    # Pisahkan fitur dan target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing untuk fitur numerik dan kategorikal
    numeric_features = ['age', 'purchase_amount', 'tenure']
    categorical_features = ['gender']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Definisikan model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluasi model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Simpan model ke file
    print("Saving model...")
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Simpan juga informasi fitur untuk inference
    feature_names = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model training completed and saved to models/churn_model.pkl")
    
    return model

if __name__ == "__main__":
    train_churn_model()