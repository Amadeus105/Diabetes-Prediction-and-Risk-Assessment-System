import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "result" in data

def test_invalid_prediction():
    response = client.post("/predict", json={
        "Pregnancies": "abc",  # Ошибка: строка вместо числа
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33
    })
    assert response.status_code == 422  # Ошибка валидации
