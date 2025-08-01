import joblib
import numpy as np
import os

def apply_custom_rules(income, debts, punctual_rate):
    """
    Applique les règles personnalisées de scoring définies dans generate_data.py
    
    Args:
        income (float): Revenu mensuel du client
        debts (float): Montant moyen des dettes
        punctual_rate (float): Taux de ponctualité (entre 0 et 1)
    
    Returns:
        int: Score de crédit selon les règles personnalisées (0 = mauvais, 1 = bon)
    """
    if punctual_rate >= 0.55 and debts / income <= 0.5:
        return 1
    elif punctual_rate >= 0.92 and income < debts:
        return 1
    else:
        return 0

def validate_input(income, debts, punctual_rate):
    """Valide les données d'entrée pour la prédiction"""
    errors = []
    
    if not isinstance(income, (int, float)) or income < 0:
        errors.append("Le revenu doit être un nombre positif")
    
    if not isinstance(debts, (int, float)) or debts < 0:
        errors.append("Les dettes doivent être un nombre positif")
    
    if not isinstance(punctual_rate, (int, float)) or punctual_rate < 0 or punctual_rate > 1:
        errors.append("Le taux de ponctualité doit être entre 0 et 1")
    
    if income == 0 and debts > 0:
        errors.append("Incohérence: revenu nul avec des dettes")
    
    if len(errors) > 0:
        raise ValueError("Erreurs de validation: " + "; ".join(errors))

def predict_credit_score(income, debts, punctual_rate):
    """
    Prédit le score de crédit d'un client
    
    Args:
        income (float): Revenu mensuel du client
        debts (float): Montant moyen des dettes
        punctual_rate (float): Taux de ponctualité (entre 0 et 1)
    
    Returns:
        int: Score de crédit (0 = mauvais, 1 = bon)
    
    Raises:
        ValueError: Si les données d'entrée sont invalides
        FileNotFoundError: Si le modèle n'est pas trouvé
    """
    validate_input(income, debts, punctual_rate)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "credit_model.pkl")
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle n'est pas trouvé à: {model_path}. "
            "Veuillez entraîner le modèle d'abord avec 'python main.py'"
        )
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Le scaler n'est pas trouvé à: {scaler_path}. "
            "Veuillez entraîner le modèle d'abord avec 'python main.py'"
        )
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_data = np.array([[income, debts, punctual_rate]])
        
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)
        
        return int(prediction[0])
        
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction: {str(e)}")

def predict_credit_score_with_proba(income, debts, punctual_rate):
    """
    Prédit le score de crédit avec les probabilités
    
    Args:
        income (float): Revenu mensuel du client
        debts (float): Montant moyen des dettes
        punctual_rate (float): Taux de ponctualité (entre 0 et 1)
    
    Returns:
        dict: Dictionnaire avec le score et les probabilités
    """
    validate_input(income, debts, punctual_rate)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "credit_model.pkl")
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Le modèle ou le scaler n'est pas encore entraîné. Veuillez entraîner le modèle d'abord.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    input_data = np.array([[income, debts, punctual_rate]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return {
        'credit_score': int(prediction[0]),
        'probability_bad': float(probabilities[0][0]),
        'probability_good': float(probabilities[0][1]),
        'confidence': float(max(probabilities[0]))
    }

def predict_with_custom_rules(income, debts, punctual_rate):
    """
    Prédit le score de crédit en combinant le modèle ML et les règles personnalisées
    
    Args:
        income (float): Revenu mensuel du client
        debts (float): Montant moyen des dettes
        punctual_rate (float): Taux de ponctualité (entre 0 et 1)
    
    Returns:
        dict: Dictionnaire avec les prédictions du modèle ML et des règles personnalisées
    """
    validate_input(income, debts, punctual_rate)
    
    custom_score = apply_custom_rules(income, debts, punctual_rate)
    
    try:
        ml_result = predict_credit_score_with_proba(income, debts, punctual_rate)
        ml_score = ml_result['credit_score']
        ml_probabilities = ml_result
    except Exception as e:
        ml_score = None
        ml_probabilities = None
    
    return {
        'custom_rules_score': custom_score,
        'ml_model_score': ml_score,
        'ml_probabilities': ml_probabilities,
        'input_data': {
            'income': income,
            'debts': debts,
            'punctual_rate': punctual_rate
        }
    }

def get_credit_interpretation(credit_score, probability_good=None):
    """
    Fournit une interprétation du score de crédit
    
    Args:
        credit_score (int): Score de crédit (0 ou 1)
        probability_good (float): Probabilité d'être un bon client
    
    Returns:
        dict: Interprétation du score
    """
    if credit_score == 1:
        interpretation = {
            'status': 'Bon client',
            'description': 'Le client présente un bon profil de crédit',
            'recommendation': 'Approuver le crédit',
            'risk_level': 'Faible'
        }
    else:
        interpretation = {
            'status': 'Client à risque',
            'description': 'Le client présente un profil de crédit risqué',
            'recommendation': 'Refuser ou demander des garanties supplémentaires',
            'risk_level': 'Élevé'
        }
    
    if probability_good is not None:
        if probability_good >= 0.8:
            confidence = 'Très élevée'
        elif probability_good >= 0.6:
            confidence = 'Élevée'
        elif probability_good >= 0.4:
            confidence = 'Modérée'
        else:
            confidence = 'Faible'
        
        interpretation['confidence'] = confidence
        interpretation['probability'] = probability_good
    
    return interpretation
