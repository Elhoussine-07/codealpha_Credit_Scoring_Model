# TASK 1: Credit Scoring Model

## Objectif

Prédire la **solvabilité** d'un individu en utilisant ses données financières passées.

##Approche

Utilisation d'algorithmes de classification comme la Régression Logistique, les Arbres de Décision et la Forêt Aléatoire.

## Fonctionnalités Clés

- **Feature Engineering** à partir de l'historique financier
- **Évaluation du modèle** avec les métriques : **Precision**, **Recall**, **F1-Score**, **ROC-AUC**
- **Dataset** incluant : revenus, dettes, historique de paiement, etc.

## Architecture du Projet

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

