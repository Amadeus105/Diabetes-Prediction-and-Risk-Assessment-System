import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Загрузка данных
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# 2. Обработка пропусков (замена 0 на NaN в некоторых колонках)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Заполнение пропусков медианами
df.fillna(df.median(), inplace=True)

# 3. Разделение данных на признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Сохранение модели и scaler
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Модель и scaler успешно сохранены.")
