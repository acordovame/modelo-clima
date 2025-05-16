from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler correctamente
modelo = load_model("modelo.keras")         # ✅ Keras model
scaler = joblib.load("scaler.pkl")          # ✅ scikit-learn scaler

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.json  # {"input": [[22.1, 64, 1013]]}
    input_array = np.array(data['input'])
    scaled = scaler.transform(input_array)
    prediction = modelo.predict(scaled)
    return jsonify(prediccion=prediction.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)