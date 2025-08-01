import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    print(f"Chargement des données depuis: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    
    required_cols = ['income', 'debts', 'punctual_rate', 'credit_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")
    
    X = df[['income', 'debts', 'punctual_rate']]
    y = df['credit_score']
    
    print("Nettoyage des données...")
    
    X = X.replace([np.inf, -np.inf], np.nan)
    
    initial_rows = len(X)
    X = X.dropna()
    y = y[X.index]
    final_rows = len(X)
    
    if initial_rows != final_rows:
        print(f"Suppression de {initial_rows - final_rows} lignes avec valeurs manquantes")
    
    print("Statistiques des features:")
    print(X.describe())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Données d'entraînement: {len(X_train)} échantillons")
    print(f"Données de test: {len(X_test)} échantillons")
    print(f"Distribution des classes (train): {np.bincount(y_train)}")
    print(f"Distribution des classes (test): {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, scaler