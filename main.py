from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Apartment Price Prediction API",
    description="API для предсказания стоимости квартиры на основе её параметров",
    version="1.0"
)

# Путь к сохраненной модели
MODEL_PATH = "model/apartment_model.pkl"

# Загружаем модель при старте приложения
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Описываем входные данные через Pydantic для валидации
class ApartmentFeatures(BaseModel):
    area: float   # Площадь в кв.м
    rooms: int    # Количество комнат
    floor: int    # Этаж

@app.get("/")
def home():
    return {"message": "Добро пожаловать! Перейдите на /docs для тестирования API."}

@app.post("/predict")
def predict_price(features: ApartmentFeatures):
    if model is None:
        return {"error": "Модель не найдена. Сначала запустите train_model.py!"}
    
    # Преобразуем входящие данные в DataFrame
    input_data = pd.DataFrame([{
        'area': features.area,
        'rooms': features.rooms,
        'floor': features.floor
    }])
    
    # Делаем предсказание
    prediction = model.predict(input_data)[0]
    
    return {
        "input_features": {
            "area": features.area,
            "rooms": features.rooms,
            "floor": features.floor
        },
        "predicted_price_k_usd": round(float(prediction), 2)
    }