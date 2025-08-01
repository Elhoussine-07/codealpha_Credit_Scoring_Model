import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from src.predict import predict_credit_score, predict_credit_score_with_proba, get_credit_interpretation, predict_with_custom_rules

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        "message": "API de Scoring de Crédit",
        "version": "1.0",
        "endpoints": {
            "/predict": "Prédiction simple (POST)",
            "/predict_detailed": "Prédiction détaillée avec probabilités (POST)",
            "/predict_custom": "Prédiction avec règles personnalisées (POST)",
            "/health": "Statut de l'API (GET)"
        },
        "usage": {
            "method": "POST",
            "content_type": "application/json",
            "parameters": {
                "income": "Revenu mensuel (float, > 0)",
                "debts": "Montant des dettes (float, >= 0)",
                "punctual_rate": "Taux de ponctualité (float, 0-1)"
            }
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Vérification de l'état de l'API"""
    try:
        test_result = predict_credit_score(50000, 20000, 0.8)
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": test_result
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de prédiction simple"""
    try:
        if not request.is_json:
            return jsonify({"error": "Le contenu doit être au format JSON"}), 400
        
        data = request.json
        
        income = data.get("income")
        debts = data.get("debts")
        punctual_rate = data.get("punctual_rate")
        
        if income is None or debts is None or punctual_rate is None:
            return jsonify({
                "error": "Paramètres manquants",
                "required": ["income", "debts", "punctual_rate"],
                "received": list(data.keys())
            }), 400
        
        credit_score = predict_credit_score(income, debts, punctual_rate)
        
        interpretation = get_credit_interpretation(credit_score)
        
        return jsonify({
            "credit_score": credit_score,
            "interpretation": interpretation,
            "input_data": {
                "income": income,
                "debts": debts,
                "punctual_rate": punctual_rate
            }
        })
        
    except ValueError as e:
        return jsonify({"error": f"Données invalides: {str(e)}"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

@app.route("/predict_detailed", methods=["POST"])
def predict_detailed():
    """Endpoint de prédiction détaillée avec probabilités"""
    try:
        if not request.is_json:
            return jsonify({"error": "Le contenu doit être au format JSON"}), 400
        
        data = request.json
        
        income = data.get("income")
        debts = data.get("debts")
        punctual_rate = data.get("punctual_rate")
        
        if income is None or debts is None or punctual_rate is None:
            return jsonify({
                "error": "Paramètres manquants",
                "required": ["income", "debts", "punctual_rate"],
                "received": list(data.keys())
            }), 400
        
        result = predict_credit_score_with_proba(income, debts, punctual_rate)
        
        interpretation = get_credit_interpretation(
            result['credit_score'], 
            result['probability_good']
        )
        
        return jsonify({
            "credit_score": result['credit_score'],
            "probabilities": {
                "bad_client": result['probability_bad'],
                "good_client": result['probability_good'],
                "confidence": result['confidence']
            },
            "interpretation": interpretation,
            "input_data": {
                "income": income,
                "debts": debts,
                "punctual_rate": punctual_rate
            }
        })
        
    except ValueError as e:
        return jsonify({"error": f"Données invalides: {str(e)}"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    """Endpoint de prédiction avec règles personnalisées"""
    try:
        if not request.is_json:
            return jsonify({"error": "Le contenu doit être au format JSON"}), 400
        
        data = request.json
        
        income = data.get("income")
        debts = data.get("debts")
        punctual_rate = data.get("punctual_rate")
        
        if income is None or debts is None or punctual_rate is None:
            return jsonify({
                "error": "Paramètres manquants",
                "required": ["income", "debts", "punctual_rate"],
                "received": list(data.keys())
            }), 400
        
        result = predict_with_custom_rules(income, debts, punctual_rate)
        
        custom_interpretation = get_credit_interpretation(result['custom_rules_score'])
        
        ml_interpretation = None
        if result['ml_model_score'] is not None:
            ml_interpretation = get_credit_interpretation(
                result['ml_model_score'], 
                result['ml_probabilities']['probability_good']
            )
        
        return jsonify({
            "custom_rules": {
                "score": result['custom_rules_score'],
                "interpretation": custom_interpretation
            },
            "ml_model": {
                "score": result['ml_model_score'],
                "probabilities": result['ml_probabilities'],
                "interpretation": ml_interpretation
            },
            "input_data": {
                "income": income,
                "debts": debts,
                "punctual_rate": punctual_rate
            }
        })
        
    except ValueError as e:
        return jsonify({"error": f"Données invalides: {str(e)}"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    """Gestion des routes non trouvées"""
    return jsonify({
        "error": "Endpoint non trouvé",
        "available_endpoints": ["/", "/predict", "/predict_detailed", "/health"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs internes"""
    return jsonify({
        "error": "Erreur interne du serveur",
        "message": "Veuillez réessayer plus tard"
    }), 500

if __name__ == '__main__':
    print("🚀 Démarrage de l'API de Scoring de Crédit...")
    print("📊 Endpoints disponibles:")
    print("   - GET  / : Documentation de l'API")
    print("   - GET  /health : Vérification de l'état")
    print("   - POST /predict : Prédiction simple")
    print("   - POST /predict_detailed : Prédiction avec probabilités")
    print("\n🌐 API accessible sur: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)