import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_and_preprocess_data

def evaluate_model(model, X_test, y_test, model_name):
    """Évalue un modèle avec toutes les métriques requises"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"\n{'='*50}")
    print(f"ÉVALUATION DU MODÈLE: {model_name}")
    print(f"{'='*50}")
    
    print("\n📊 RAPPORT DE CLASSIFICATION:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 MATRICE DE CONFUSION:")
    print(cm)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()
    
    return {
        'model_name': model_name,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm,
        'roc_auc': roc_auc if y_pred_proba is not None else None
    }

def compare_models(X_train, X_test, y_train, y_test):
    """Compare plusieurs algorithmes de classification"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Entraînement de {name}...")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        model.fit(X_train, y_train)
        
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    return results

def optimize_random_forest(X_train, y_train):
    """Optimise les hyperparamètres du Random Forest"""
    print("\n🔧 OPTIMISATION DES HYPERPARAMÈTRES - RANDOM FOREST")
    print("="*60)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "custom_credit_data.csv")
    models_dir = os.path.join(project_root, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Chargement des données depuis: {data_path}")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    print("\n🚀 COMPARAISON DES MODÈLES")
    print("="*50)
    results = compare_models(X_train, X_test, y_train, y_test)
    
    best_model = optimize_random_forest(X_train, y_train)
    
    final_results = evaluate_model(best_model, X_test, y_test, "Random Forest (Optimisé)")
    
    model_path = os.path.join(models_dir, "credit_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ Modèle optimisé enregistré dans: {model_path}")
    print(f"✅ Scaler enregistré dans: {scaler_path}")
    
    results_path = os.path.join(models_dir, "evaluation_results.pkl")
    joblib.dump(results, results_path)
    print(f"✅ Résultats d'évaluation enregistrés dans: {results_path}")
    
    print("\n📋 RÉSUMÉ DES PERFORMANCES")
    print("="*50)
    for name, result in results.items():
        if result['roc_auc']:
            print(f"{name}: ROC-AUC = {result['roc_auc']:.4f}")
    
    print("Entraînement terminé avec succès!")

if __name__ == "__main__":
    train_model()