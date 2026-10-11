import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


data = {
    'area': [40, 55, 70, 85, 100, 45, 60, 80, 95, 110],
    'rooms': [1, 2, 2, 3, 4, 1, 2, 3, 3, 4],
    'floor': [3, 5, 2, 9, 4, 12, 7, 1, 5, 8],
    'price': [60, 85, 105, 130, 160, 70, 90, 125, 140, 175]
}

df = pd.DataFrame(data)


X = df[['area', 'rooms', 'floor']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/apartment_model.pkl')

print("Модель успешно обучена и сохранена в model/apartment_model.pkl!")