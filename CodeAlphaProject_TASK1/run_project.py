#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_header(title):
    """Affiche un en-tête stylisé"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Affiche une étape du processus"""
    print(f"\n📋 Étape {step}: {description}")
    print("-" * 40)

def check_file_exists(file_path):
    """Vérifie si un fichier existe"""
    return Path(file_path).exists()

def run_command(command, description, check_output=False):
    """Exécute une commande avec gestion d'erreur"""
    print(f"🔄 {description}...")
    try:
        if check_output:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {description} terminé avec succès")
            return result.stdout
        else:
            subprocess.run(command, shell=True, check=True)
            print(f"✅ {description} terminé avec succès")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de {description}: {e}")
        return False

def wait_for_api(url, max_attempts=30):
    """Attend que l'API soit disponible"""
    print(f"⏳ Attente de l'API sur {url}...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("✅ API prête!")
                return True
        except:
            pass
        time.sleep(1)
        if attempt % 5 == 0:
            print(f"   Tentative {attempt + 1}/{max_attempts}...")
    
    print("❌ L'API n'est pas accessible après le délai d'attente")
    return False

def main():
    """Fonction principale"""
    print_header("CREDIT SCORING MODEL - LANCEMENT COMPLET")
    
    print_step(1, "Vérification de l'environnement")
    
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ requis")
        return False
    
    if not check_file_exists("requirements.txt"):
        print("❌ Fichier requirements.txt manquant")
        return False
    
    print_step(2, "Installation des dépendances")
    if not run_command("pip install -r requirements.txt", "Installation des packages"):
        return False
    
    print_step(3, "Génération des données")
    if not check_file_exists("data/custom_credit_data.csv"):
        if not run_command("python generate_data.py", "Génération des données d'entraînement"):
            return False
    else:
        print("✅ Données déjà générées")
    
    print_step(4, "Entraînement du modèle")
    if not check_file_exists("models/credit_model.pkl"):
        if not run_command("python main.py", "Entraînement du modèle"):
            return False
    else:
        print("✅ Modèle déjà entraîné")
    
    print_step(5, "Tests unitaires")
    if not run_command("python -m pytest tests/ -v", "Exécution des tests"):
        print("⚠️ Tests échoués, mais continuation...")
    
    print_step(6, "Démarrage de l'API Flask")
    api_process = subprocess.Popen(
        ["python", "app/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if not wait_for_api("http://127.0.0.1:5000/health"):
        print("❌ Impossible de démarrer l'API")
        api_process.terminate()
        return False
    
    print_step(7, "Test de l'API")
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"income": 50000, "debts": 20000, "punctual_rate": 0.8},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Test API réussi - Score: {result['credit_score']}")
        else:
            print(f"⚠️ Test API échoué - Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Erreur lors du test API: {e}")
    
    print_header("🎉 PROJET PRÊT!")
    print("""
📊 Interface Streamlit disponible sur: http://localhost:8501
🌐 API Flask disponible sur: http://127.0.0.1:5000

📋 Commandes utiles:
   • Interface: streamlit run interface/streamlit_app.py
   • API: python app/app.py
   • Tests: python -m pytest tests/ -v
   • Entraînement: python main.py

🛑 Pour arrêter: Ctrl+C
""")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du projet...")
        api_process.terminate()
        print("✅ Projet arrêté proprement")

if __name__ == "__main__":
    main() 