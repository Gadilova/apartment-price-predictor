"""Обучение модели: читаем данные о квартирах, обучаем линейную регрессию и сохраняем её в файл.

Запуск из корня проекта:
    python model/train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = "data/housing.csv"
MODEL_PATH = "model/apartment_model.pkl"

FEATURES = ["area", "rooms", "floor"]
TARGET = "price"  # цена в тысячах долларов


def train():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    model = LinearRegression()
    model.fit(X, y)

    # R² показывает, насколько хорошо модель объясняет цены: 1.0 — идеально, 0 — никак
    print(f"Обучено на {len(df)} квартирах, R² = {model.score(X, y):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    train()
