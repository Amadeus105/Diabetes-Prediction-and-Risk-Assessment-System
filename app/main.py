from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.schemas import PatientData
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction API")

# Загружаем модель и scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Монтируем статику в /static (чтобы не конфликтовать с API)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.post("/predict")
def predict(data: PatientData):
    X = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "result": "Diabetic" if prediction == 1 else "Not diabetic"
    }
