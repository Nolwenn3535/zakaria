import logging
from flask import Flask, render_template, request
import pickle
import pandas as pd
from arize.pandas.logger import Client, Schema, Environments, ModelTypes
import os
from dotenv import load_dotenv

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Initialiser le client Arize
arize_client = Client(
    api_key=os.getenv('ARIZE_API_KEY'),
    space_key=os.getenv('ARIZE_PROJECT_ID')
)

# Définir le schéma des données pour Arize
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=[
        "credit_lines_outstanding", 
        "loan_amt_outstanding", 
        "total_debt_outstanding", 
        "income", 
        "years_employed", 
        "fico_score"
    ],
    prediction_label_column_name="prediction"
)

# Initialiser l'application Flask et charger le modèle
app = Flask(__name__)
model = pickle.load(open("random_forest_model2.pkl", "rb"))

# Fonction pour effectuer une prédiction
def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    logger.info(f"Prediction made: {prediction[0]} for features: {features}")
    return int(prediction[0])

# Fonction pour enregistrer les prédictions avec Arize
def log_prediction(features, prediction):
    log_data = pd.DataFrame([{
        "credit_lines_outstanding": features["credit_lines_outstanding"],
        "loan_amt_outstanding": features["loan_amt_outstanding"],
        "total_debt_outstanding": features["total_debt_outstanding"],
        "income": features["income"],
        "years_employed": features["years_employed"],
        "fico_score": features["fico_score"],
        "prediction": prediction,
        "prediction_id": str(pd.util.hash_pandas_object(pd.Series(features)).values[0]),  # Générer un ID unique pour chaque prédiction
        "timestamp": pd.Timestamp.now()
    }])

    try:
        # Envoyer les données à Arize
        response = arize_client.log(
            dataframe=log_data,
            schema=schema,
            environment=Environments.PRODUCTION,  # Définir l'environnement à 'Production'
            model_id="random_forest_model2",
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            model_version="1.0",
            validate=True
        )
        logger.info("Successfully logged data to Arize. Response:",response)
    except Exception as e:
        logger.error("Error logging data to Arize: ",e)

# Page d'accueil
@app.route("/", methods=["GET"])
def Home():
    logger.info("Home page accessed.")
    return render_template("index.html")

# Route pour la prédiction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extraire les données du formulaire
        credit_lines_outstanding = int(request.form["lignes de crédit en cours"])
        loan_amt_outstanding = int(request.form["montant du prêt en cours"])
        total_debt_outstanding = int(request.form["dette totale en cours"])
        income = float(request.form["revenu"])
        years_employed = int(request.form["années d'emploi"])
        fico_score = int(request.form["score FICO"])

        # Préparer les données pour la prédiction
        features = {
            "credit_lines_outstanding": credit_lines_outstanding,
            "loan_amt_outstanding": loan_amt_outstanding,
            "total_debt_outstanding": total_debt_outstanding,
            "income": income,
            "years_employed": years_employed,
            "fico_score": fico_score,
        }

        logger.info("Received features: ",features)

        # Effectuer la prédiction
        prediction = model_pred(features)

        # Déterminer le texte du résultat
        if prediction == 1:
            prediction_text = "Le client est à risque de défaut de paiement."
        else:
            prediction_text = "Le client n'est pas à risque de défaut de paiement."

        logger.info("Prediction result : ",prediction_text)

        # Enregistrer la prédiction avec Arize
        log_prediction(features, prediction)

        # Renvoyer la page avec le résultat de la prédiction
        return render_template("index.html", prediction_text=prediction_text)
    else:
        logger.info("Non-POST request made to /predict route.")
        return render_template("index.html")

# Exécuter l'application Flask
if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
