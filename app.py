from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

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
        data = request.get_json(force=True)
        entrada = np.array(data['input'])  # Esperado: (1, 24, 3)

        if entrada.shape != (1, 24, 3):
            return jsonify({'error': f'Forma inválida: {entrada.shape}. Se esperaba (1, 24, 3).'}), 400

        # Escalar
        entrada_2d = entrada.reshape(-1, 3)  # (24, 3)
        entrada_scaled = scaler.transform(entrada_2d)
        entrada_scaled = entrada_scaled.reshape(1, 24, 3)

        # Predecir
        pred_scaled = modelo.predict(entrada_scaled)
        pred = scaler.inverse_transform(pred_scaled)

        return jsonify({'prediccion': pred.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Requiere esto en Render
    app.run(host="0.0.0.0", port=port)
