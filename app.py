from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carga el modelo
modelo = joblib.load("modelo.pkl")

@app.route("/")
def home():
    return "API de Machine Learning funcionando"

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Faltan datos"}), 400

    try:
        prediccion = modelo.predict([np.array(data["features"])])
        return jsonify({"resultado": prediccion[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)