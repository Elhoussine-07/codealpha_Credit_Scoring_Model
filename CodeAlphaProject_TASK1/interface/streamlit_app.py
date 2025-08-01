import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="🏦 Credit Scoring Model",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Vérifie l'état de l'API"""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def predict_credit_score(income, debts, punctual_rate, detailed=False):
    """Fait une prédiction via l'API"""
    endpoint = "/predict_detailed" if detailed else "/predict"
    try:
        response = requests.post(
            f"http://127.0.0.1:5000{endpoint}",
            json={
                "income": income,
                "debts": income,
                "punctual_rate": punctual_rate
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except requests.exceptions.ConnectionError:
        return False, {"error": "Impossible de se connecter à l'API Flask"}
    except Exception as e:
        return False, {"error": f"Erreur: {str(e)}"}

def predict_with_custom_rules(income, debts, punctual_rate):
    """Fait une prédiction avec règles personnalisées via l'API"""
    try:
        response = requests.post(
            f"http://127.0.0.1:5000/predict_custom",
            json={
                "income": income,
                "debts": debts,
                "punctual_rate": punctual_rate
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except requests.exceptions.ConnectionError:
        return False, {"error": "Impossible de se connecter à l'API Flask"}
    except Exception as e:
        return False, {"error": f"Erreur: {str(e)}"}

def create_credit_score_gauge(score, probability_good=None):
    """Crée un graphique en jauge pour le score de crédit"""
    if probability_good is not None:
        value = probability_good * 100
        color = "green" if score == 1 else "red"
    else:
        value = 100 if score == 1 else 0
        color = "green" if score == 1 else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Crédit"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">🏦 Credit Scoring Model</h1>', unsafe_allow_html=True)
    
    api_healthy, api_status = check_api_health()
    
    if not api_healthy:
        st.error("🚫 L'API Flask n'est pas accessible. Veuillez la démarrer avec :")
        st.code("cd app && python app.py")
        st.stop()
    
    st.sidebar.title("⚙️ Paramètres")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prédiction", "📈 Analyse", "📋 Historique", "ℹ️ À propos"])
    
    with tab1:
        st.header("🎯 Prédiction du Score de Crédit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input(
                "💰 Revenu mensuel (€)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=1000,
                help="Revenu mensuel net du client"
            )
            
            debts = st.number_input(
                "💳 Montant des dettes (€)",
                min_value=0,
                max_value=1000000,
                value=20000,
                step=1000,
                help="Montant total des dettes actuelles"
            )
        
        with col2:
            punctual_rate = st.slider(
                "⏰ Taux de ponctualité",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.01,
                help="Proportion de paiements effectués à temps (0 = jamais, 1 = toujours)"
            )
            
            st.metric("Ratio Dette/Revenu", f"{(debts/income)*100:.1f}%" if income > 0 else "N/A")
            st.metric("Taux de Ponctualité", f"{punctual_rate*100:.0f}%")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Prédire le Score", type="primary", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    custom_success, custom_result = predict_with_custom_rules(income, debts, punctual_rate)
                    ml_success, ml_result = predict_credit_score(income, debts, punctual_rate, detailed=True)
                
                if custom_success and ml_success:
                    st.success("✅ Prédictions effectuées avec succès!")
                    
                    st.subheader("🔄 Comparaison des Prédictions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📋 Règles Personnalisées")
                        custom_score = custom_result['custom_rules']['score']
                        custom_interp = custom_result['custom_rules']['interpretation']
                        
                        st.metric(
                            "Score (Règles)",
                            "✅ Bon" if custom_score == 1 else "❌ Risqué",
                            delta="Règles directes"
                        )
                        
                        if custom_score == 1:
                            st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        **Statut:** {custom_interp['status']}  
                        **Niveau de risque:** {custom_interp['risk_level']}  
                        **Recommandation:** {custom_interp['recommendation']}
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Règles appliquées:**")
                        if punctual_rate >= 0.55 and debts/income <= 0.5:
                            st.markdown("✅ `punctual_rate >= 0.55 AND debts/income <= 0.5`")
                        elif punctual_rate >= 0.92 and income < debts:
                            st.markdown("✅ `punctual_rate >= 0.92 AND income < debts`")
                        else:
                            st.markdown("❌ Aucune condition remplie → Score = 0")
                    
                    with col2:
                        st.markdown("### 🤖 Modèle Machine Learning")
                        ml_score = ml_result['credit_score']
                        ml_proba = ml_result['probabilities']
                        ml_interp = ml_result['interpretation']
                        
                        st.metric(
                            "Score (ML)",
                            "✅ Bon" if ml_score == 1 else "❌ Risqué",
                            delta=f"{ml_proba['confidence']*100:.1f}% confiance"
                        )
                        
                        st.metric(
                            "Probabilité Bon Client",
                            f"{ml_proba['good_client']*100:.1f}%"
                        )
                        
                        if ml_score == 1:
                            st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        **Statut:** {ml_interp['status']}  
                        **Niveau de risque:** {ml_interp['risk_level']}  
                        **Recommandation:** {ml_interp['recommendation']}  
                        **Confiance:** {ml_interp.get('confidence', 'N/A')}
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.subheader("📊 Visualisation du Modèle ML")
                    st.plotly_chart(create_credit_score_gauge(
                        ml_score,
                        ml_proba['good_client']
                    ), use_container_width=True)
                    
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'income': income,
                        'debts': debts,
                        'punctual_rate': punctual_rate,
                        'custom_score': custom_score,
                        'ml_score': ml_score,
                        'probability_good': ml_proba['good_client'],
                        'interpretation': ml_interp
                    })
                    
                elif custom_success:
                    st.warning("⚠️ Seules les règles personnalisées sont disponibles")
                    custom_score = custom_result['custom_rules']['score']
                    custom_interp = custom_result['custom_rules']['interpretation']
                    
                    st.metric(
                        "Score (Règles)",
                        "✅ Bon" if custom_score == 1 else "❌ Risqué",
                        delta="Règles directes"
                    )
                    
                    if custom_score == 1:
                        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    **Statut:** {custom_interp['status']}  
                    **Niveau de risque:** {custom_interp['risk_level']}  
                    **Recommandation:** {custom_interp['recommendation']}
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ Erreur: {custom_result.get('error', 'Erreur inconnue')}")
    
    with tab2:
        st.header("📈 Analyse des Données")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution des Scores")
                
                if 'ml_score' in df.columns:
                    fig_ml = px.pie(
                        df, 
                        names='ml_score', 
                        title="Répartition des Scores ML",
                        labels={'0': 'Risqué', '1': 'Bon'}
                    )
                    st.plotly_chart(fig_ml, use_container_width=True)
                
                if 'custom_score' in df.columns:
                    fig_custom = px.pie(
                        df, 
                        names='custom_score', 
                        title="Répartition des Scores Règles",
                        labels={'0': 'Risqué', '1': 'Bon'}
                    )
                    st.plotly_chart(fig_custom, use_container_width=True)
            
            with col2:
                st.subheader("Évolution des Probabilités")
                if len(st.session_state.prediction_history) > 1 and 'probability_good' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    fig = px.line(
                        df, 
                        x='timestamp', 
                        y='probability_good',
                        title="Probabilité de Bon Client au Fil du Temps"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'custom_score' in df.columns and 'ml_score' in df.columns:
                    st.subheader("Comparaison des Scores")
                    
                    comparison_data = []
                    for idx, row in df.iterrows():
                        comparison_data.append({
                            'timestamp': row['timestamp'],
                            'Score': 'Règles',
                            'Valeur': row['custom_score']
                        })
                        comparison_data.append({
                            'timestamp': row['timestamp'],
                            'Score': 'ML',
                            'Valeur': row['ml_score']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'])
                    
                    fig_comparison = px.line(
                        comparison_df,
                        x='timestamp',
                        y='Valeur',
                        color='Score',
                        title="Comparaison des Scores au Fil du Temps",
                        labels={'Valeur': 'Score (0=Risqué, 1=Bon)'}
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("Aucune donnée d'historique disponible. Effectuez votre première prédiction!")
    
    with tab3:
        st.header("📋 Historique des Prédictions")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            
            display_columns = ['date', 'time', 'income', 'debts', 'punctual_rate']
            column_mapping = {
                'date': 'Date',
                'time': 'Heure',
                'income': 'Revenu',
                'debts': 'Dettes',
                'punctual_rate': 'Ponctualité'
            }
            
            if 'custom_score' in df.columns:
                display_columns.append('custom_score')
                column_mapping['custom_score'] = 'Score Règles'
            
            if 'ml_score' in df.columns:
                display_columns.append('ml_score')
                column_mapping['ml_score'] = 'Score ML'
            
            if 'probability_good' in df.columns:
                display_columns.append('probability_good')
                column_mapping['probability_good'] = 'Prob. Bon'
            
            st.dataframe(
                df[display_columns].rename(columns=column_mapping),
                use_container_width=True
            )
            
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("Aucune prédiction dans l'historique. Effectuez votre première prédiction!")
    
    with tab4:
        st.header("ℹ️ À propos du Modèle")
        
        st.markdown("""
        ### 🎯 Objectif
        Ce modèle prédit la **solvabilité** d'un client en utilisant ses données financières.
        
        ### 🔄 Deux Approches de Scoring
        
        #### 📋 Règles Personnalisées
        **Logique métier directe** basée sur des critères explicites :
        
        ```python
        if punctual_rate >= 0.55 and debts/income <= 0.5:
            return 1  # Bon client
        elif punctual_rate >= 0.92 and income < debts:
            return 1  # Bon client (exception)
        else:
            return 0  # Client à risque
        ```
        
        **Interprétation :**
        - **Condition 1** : Ponctualité ≥ 55% ET ratio dette/revenu ≤ 50%
        - **Condition 2** : Ponctualité ≥ 92% (excellente) même si revenu < dettes
        - **Sinon** : Client à risque
        
        #### 🤖 Modèle Machine Learning
        **Random Forest optimisé** qui a appris à partir des données générées par les règles personnalisées :
        
        - **Régression Logistique** : Modèle linéaire interprétable
        - **Arbre de Décision** : Modèle non-linéaire simple
        - **Forêt Aléatoire** ⭐ : Modèle final optimisé
        
        ### 🎯 Métriques d'Évaluation
        - **Precision** : Éviter les faux positifs
        - **Recall** : Ne pas manquer de bons clients
        - **F1-Score** : Équilibre entre precision et recall
        - **ROC-AUC** : Performance globale du modèle
        
        ### 📈 Features Utilisées
        - **Revenu mensuel** : Capacité de remboursement
        - **Dettes actuelles** : Charge financière
        - **Taux de ponctualité** : Historique de paiement
        
        ### 🔧 Optimisation
        Le modèle Random Forest est optimisé avec GridSearchCV pour maximiser le ROC-AUC.
        """)
        
        st.subheader("🔧 Statut de l'API")
        if api_healthy:
            st.success("✅ API opérationnelle")
            st.json(api_status)
        else:
            st.error("❌ API non accessible")

if __name__ == "__main__":
    main()
