from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

# Описываем входные данные через Pydantic для валидации.
# Field задаёт границы: FastAPI сам вернёт ошибку 422, если значение вне диапазона
class ApartmentFeatures(BaseModel):
    area: float = Field(gt=0, le=1000, description="Площадь в кв.м")
    rooms: int = Field(ge=1, le=20, description="Количество комнат")
    floor: int = Field(ge=1, le=200, description="Этаж")

@app.get("/")
def home():
    return {"message": "Добро пожаловать! Перейдите на /docs для тестирования API."}

@app.post("/predict")
def predict_price(features: ApartmentFeatures):
    if model is None:
        # 503 — «сервис пока недоступен»: модели нет, предсказывать нечем
        raise HTTPException(status_code=503, detail="Модель не найдена. Сначала запустите python model/train_model.py")
    
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
