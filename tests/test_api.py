"""Тесты API: обучаем модель, отправляем запросы и проверяем ответы."""

from fastapi.testclient import TestClient

import main
from model.train_model import train

train()  # перед тестами обучаем свежую модель на data/housing.csv
main.model = main.joblib.load(main.MODEL_PATH)
client = TestClient(main.app)


def test_home():
    assert client.get("/").status_code == 200


def test_predict_returns_price():
    response = client.post("/predict", json={"area": 70, "rooms": 2, "floor": 5})
    assert response.status_code == 200
    assert 80 < response.json()["predicted_price_k_usd"] < 130


def test_bigger_apartment_costs_more():
    small = client.post("/predict", json={"area": 40, "rooms": 1, "floor": 3}).json()
    big = client.post("/predict", json={"area": 110, "rooms": 4, "floor": 3}).json()
    assert big["predicted_price_k_usd"] > small["predicted_price_k_usd"]


def test_invalid_input():
    assert client.post("/predict", json={"area": -5, "rooms": 2, "floor": 1}).status_code == 422
    assert client.post("/predict", json={"area": 50, "rooms": 0, "floor": 1}).status_code == 422
