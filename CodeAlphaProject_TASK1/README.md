# 🏦 TASK 1: Credit Scoring Model

## 📋 Objectif

Prédire la **solvabilité** d'un individu en utilisant ses données financières passées.

## 🎯 Approche

Utilisation d'**algorithmes de classification** comme la Régression Logistique, les Arbres de Décision et la Forêt Aléatoire.

## ✨ Fonctionnalités Clés

- **Feature Engineering** à partir de l'historique financier
- **Évaluation du modèle** avec les métriques : **Precision**, **Recall**, **F1-Score**, **ROC-AUC**
- **Dataset** incluant : revenus, dettes, historique de paiement, etc.

## 🏗️ Architecture du Projet

```
CodeAlphaProject_TASK1/
├── 📁 app/                    # API Flask
│   └── app.py                # Serveur API pour prédictions
├── 📁 data/                   # Données
│   ├── custom_credit_data.csv # Données prétraitées
│   └── default of credit card clients.xls # Données brutes UCI
├── 📁 interface/              # Interface utilisateur
│   └── streamlit_app.py      # Application Streamlit
├── 📁 models/                 # Modèles entraînés
│   ├── credit_model.pkl      # Modèle optimisé
│   ├── scaler.pkl           # StandardScaler
│   └── evaluation_results.pkl # Résultats d'évaluation
├── 📁 notebooks/              # Notebooks d'analyse
│   └── EDA.ipynb            # Analyse exploratoire
├── 📁 plots/                  # Graphiques générés
├── 📁 src/                    # Code source
│   ├── data_preprocessing.py # Prétraitement des données
│   ├── train_model.py       # Entraînement et évaluation
│   └── predict.py           # Fonctions de prédiction
├── 📁 tests/                  # Tests unitaires
│   └── test_model.py        # Tests complets
├── generate_data.py          # Génération des données
├── main.py                   # Point d'entrée principal
└── requirements.txt          # Dépendances
```

## 🚀 Installation et Utilisation

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Génération des données

```bash
python generate_data.py
```

### 3. Entraînement du modèle

```bash
python main.py
```

### 4. Lancement de l'API Flask

```bash
cd app
python app.py
```

### 5. Lancement de l'interface Streamlit

```bash
cd interface
streamlit run streamlit_app.py
```

## 📊 Algorithmes Implémentés

### 1. **Régression Logistique**

- Modèle linéaire pour classification binaire
- Interprétabilité élevée
- Rapide à entraîner

### 2. **Arbre de Décision**

- Modèle non-linéaire
- Facilement interprétable
- Peut capturer des relations complexes

### 3. **Forêt Aléatoire** ⭐ (Modèle final)

- Ensemble de plusieurs arbres
- Excellente performance
- Robustesse aux overfitting

## 🎯 Métriques d'Évaluation

### **Precision**

- Proportion de prédictions positives correctes
- Important pour éviter les faux positifs

### **Recall**

- Proportion de vrais positifs détectés
- Important pour ne pas manquer de bons clients

### **F1-Score**

- Moyenne harmonique entre Precision et Recall
- Métrique équilibrée

### **ROC-AUC**

- Aire sous la courbe ROC
- Mesure globale de performance (0.5 = aléatoire, 1.0 = parfait)

## 📈 Features Utilisées

| Feature         | Description                       | Type            |
| --------------- | --------------------------------- | --------------- |
| `income`        | Revenu mensuel du client          | Numérique       |
| `debts`         | Montant moyen des dettes          | Numérique       |
| `punctual_rate` | Taux de ponctualité des paiements | Numérique (0-1) |

## 🔧 Optimisation des Hyperparamètres

Le modèle Random Forest est optimisé avec GridSearchCV sur :

- `n_estimators`: [50, 100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

## Tests

Exécuter les tests unitaires :

```bash
python -m pytest tests/
```

Ou avec unittest :

```bash
python tests/test_model.py
```

## 📱 Interface Utilisateur

### API REST (Flask)

- **Endpoint**: `POST /predict`
- **Paramètres**: `income`, `debts`, `punctual_rate`
- **Retour**: `credit_score` (0 ou 1)

### Interface Web (Streamlit)

- Interface intuitive avec sliders et inputs
- Affichage en temps réel des prédictions
- Validation des données d'entrée

## 📊 Exemple d'Utilisation

```python
from src.predict import predict_credit_score

# Exemple de prédiction
score = predict_credit_score(
    income=50000,      # Revenu mensuel
    debts=20000,       # Dettes moyennes
    punctual_rate=0.9  # Taux de ponctualité
)
print(f"Score de crédit: {score}")  # 0 = mauvais, 1 = bon
```

## 🔍 Analyse des Performances

Le modèle génère automatiquement :

- **Rapports de classification** détaillés
- **Matrices de confusion** visuelles
- **Courbes ROC** pour chaque algorithme
- **Comparaison des performances** entre modèles

## 📝 Logique Métier

Le score de crédit est calculé selon la logique :

```python
def compute_score(row):
    if row["punctual_rate"] >= 0.55 and row["debts"] / row["income"] <= 0.5:
        return 1  # Bon client
    elif row["punctual_rate"] >= 0.92 and row["income"] < row["debts"]:
        return 1  # Exception pour très ponctuel
    else:
        return 0  # Client à risque
```

## 🎯 Résultats Attendus

- **ROC-AUC > 0.8** pour un bon modèle
- **F1-Score > 0.7** pour un équilibre Precision/Recall
- **Cross-validation** pour éviter l'overfitting

## 🔄 Workflow Complet

1. **Génération des données** → `generate_data.py`
2. **Prétraitement** → `src/data_preprocessing.py`
3. **Entraînement & Évaluation** → `src/train_model.py`
4. **Tests** → `tests/test_model.py`
5. **Déploiement** → `app/app.py` + `interface/streamlit_app.py`

## 📚 Dépendances Principales

- `scikit-learn` : Algorithmes de ML
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques
- `matplotlib` : Visualisations
- `streamlit` : Interface web
- `flask` : API REST
- `joblib` : Sauvegarde des modèles

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Ajouter des tests
4. Soumettre une pull request

## 📄 Licence

Ce projet est développé dans le cadre de l'internship CodeAlpha.

---

**Auteur**: [Votre nom]  
**Date**: 2024  
**Version**: 1.0
