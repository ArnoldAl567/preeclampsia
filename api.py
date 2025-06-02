from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("modelo_preeclampsia.joblib")

@app.route("/predict", methods=["POST"])
def predecir():
    datos = request.json
    try:
        # Asegúrate de que los valores estén en el orden correcto
        entrada = np.array([[  
            datos["embarazos"],
            datos["partos_viables"],
            datos["edad_gestacional"],
            datos["edad"],
            datos["imc"],
            datos["diabetes"],
            datos["hipertension"],
            datos["presion_sistolica"],
            datos["presion_diastolica"],
            datos["hemoglobina"],
            datos["peso_fetal"],
            datos["liquido_amnio"]
        ]])


        prediccion = modelo.predict(entrada)
        return jsonify({"riesgo": int(prediccion[0])})  # Puedes devolver texto o etiquetas si prefieres

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
