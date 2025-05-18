import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Загрузка данных
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Заменяем нули на NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Заполняем пропуски медианами
df.fillna(df.median(), inplace=True)

# Подготовка данных
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохраняем
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Модель и scaler успешно сохранены.")
