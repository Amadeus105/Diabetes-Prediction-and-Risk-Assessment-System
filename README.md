# Diabetes Prediction API

This project is a FastAPI-based machine learning API that predicts whether a person has diabetes based on medical parameters.

## Features

- Predict diabetes presence using logistic regression model
- Input patient medical data (Glucose, BloodPressure, BMI, etc.)
- Returns prediction, probability, and human-readable result
- Model trained on publicly available diabetes dataset

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
Create and activate virtual environment (recommended):

bash
Копировать
Редактировать
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies:

bash
Копировать
Редактировать
pip install -r requirements.txt
Running the API
Start the FastAPI server:

bash
Копировать
Редактировать
uvicorn app.main:app --reload
The API will be available at: http://127.0.0.1:8000

Usage
Send POST requests to /predict endpoint with patient data in JSON format:

Example JSON body:

json
Копировать
Редактировать
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 33
}
Response
json
Копировать
Редактировать
{
  "prediction": 1,
  "probability": 0.85,
  "result": "Diabetic"
}
Deployment
You can deploy this API to platforms like Render, Heroku, or others.
Make sure to upload your models/ folder with saved model and scaler files.

License
MIT License
