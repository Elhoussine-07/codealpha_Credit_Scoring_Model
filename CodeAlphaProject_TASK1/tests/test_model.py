import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_and_preprocess_data
from src.predict import predict_credit_score
from src.train_model import evaluate_model, compare_models

class TestCreditModel(unittest.TestCase):
    
    def setUp(self):
        """Configuration initiale pour les tests"""
        np.random.seed(42)
        n_samples = 1000
        
        self.test_data = pd.DataFrame({
            'income': np.random.normal(50000, 20000, n_samples),
            'debts': np.random.normal(20000, 10000, n_samples),
            'punctual_rate': np.random.uniform(0.5, 1.0, n_samples),
            'credit_score': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        self.test_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.test_scaler = StandardScaler()
        
        X = self.test_data[['income', 'debts', 'punctual_rate']].values
        y = self.test_data['credit_score'].values
        
        X_scaled = self.test_scaler.fit_transform(X)
        self.test_model.fit(X_scaled, y)
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    def test_data_preprocessing(self):
        """Test du prétraitement des données"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(temp_file)
            
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            self.assertIsNotNone(y_train)
            self.assertIsNotNone(y_test)
            self.assertIsNotNone(scaler)
            
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(X_train.shape[1], 3)
            
            self.assertAlmostEqual(X_train.mean(), 0, places=1)
            self.assertAlmostEqual(X_train.std(), 1, places=1)
            
        finally:
            os.unlink(temp_file)
    
    def test_predict_credit_score(self):
        """Test de la fonction de prédiction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            scaler_path = os.path.join(temp_dir, "test_scaler.pkl")
            
            joblib.dump(self.test_model, model_path)
            joblib.dump(self.test_scaler, scaler_path)
            
            prediction = predict_credit_score(50000, 20000, 0.8)
            
            self.assertIn(prediction, [0, 1])
    
    def test_evaluate_model(self):
        """Test de l'évaluation du modèle"""
        results = evaluate_model(self.test_model, self.X_test, self.y_test, "Test Model")
        
        self.assertIn('model_name', results)
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('roc_auc', results)
        
        if results['roc_auc'] is not None:
            self.assertGreaterEqual(results['roc_auc'], 0)
            self.assertLessEqual(results['roc_auc'], 1)
    
    def test_compare_models(self):
        """Test de la comparaison des modèles"""
        results = compare_models(self.X_train, self.X_test, self.y_train, self.y_test)
        
        self.assertIn('Logistic Regression', results)
        self.assertIn('Decision Tree', results)
        self.assertIn('Random Forest', results)
        
        for model_name, result in results.items():
            self.assertIn('roc_auc', result)
            self.assertIn('classification_report', result)
            self.assertIn('confusion_matrix', result)
    
    def test_model_predictions_range(self):
        """Test que les prédictions sont dans la plage attendue"""
        test_cases = [
            (30000, 15000, 0.9),
            (80000, 50000, 0.6),
            (20000, 25000, 0.3),
        ]
        
        for income, debts, punctual_rate in test_cases:
            prediction = predict_credit_score(income, debts, punctual_rate)
            self.assertIn(prediction, [0, 1], 
                         f"Prédiction invalide pour income={income}, debts={debts}, punctual_rate={punctual_rate}")
    
    def test_data_validation(self):
        """Test de validation des données d'entrée"""
        with self.assertRaises(Exception):
            predict_credit_score(-1000, 20000, 0.8)
        
        with self.assertRaises(Exception):
            predict_credit_score(50000, 20000, 1.5)
    
    def test_model_persistence(self):
        """Test de sauvegarde et chargement du modèle"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")
            
            joblib.dump(self.test_model, model_path)
            joblib.dump(self.test_scaler, scaler_path)
            
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(scaler_path))
            
            loaded_model = joblib.load(model_path)
            loaded_scaler = joblib.load(scaler_path)
            
            test_input = np.array([[50000, 20000, 0.8]])
            test_scaled = loaded_scaler.transform(test_input)
            prediction = loaded_model.predict(test_scaled)
            
            self.assertIn(prediction[0], [0, 1])

if __name__ == '__main__':
    unittest.main(verbosity=2)
