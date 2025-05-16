from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
modelo = load_model("modelo.keras")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return "✅ API para predicción del clima con LSTM está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.json  # Ejemplo: {"input": [[22.1, 64.2, 1012.3]]}
        entrada = np.array(data['input'])  # debe ser una lista de listas
        entrada_esc = scaler.transform(entrada)
        pred = modelo.predict(entrada_esc)
        pred_inv = scaler.inverse_transform(pred)
        return jsonify({'prediccion': pred_inv.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)