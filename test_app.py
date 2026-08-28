import pytest
from httpx import AsyncClient
from API_motor import app  # Импортируем наше приложение FastAPI из файла API_motor
from httpx import ASGITransport


@pytest.mark.asyncio
async def test_predict_endpoint_success():
    """
    Тест проверяет, что наш эндпоинт /predict работает успешно (код 200)
    и возвращает корректную структуру данных и физически адекватный результат.
    """
    # Создаем виртуального асинхронного клиента для отправки запросов внутрь FastAPI
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Отправляем тестовый запрос (скорость 1500, температура 300)
        response = await ac.get("/predict?speed=1500&temp=300")

    # Проверка 1: Сервер должен ответить статусом 200 OK (запрос прошел успешно)
    assert response.status_code == 200

    # Превращаем ответ сервера обратно в питоновский словарь
    data = response.json()

    # Проверка 2: В ответе должен быть статус "success"
    assert data["status"] == "success"

    # Проверка 3 3: Проверяем, что возвращаемое число крутящего момента физически адекватно
    # При 1500 об/мин момент должен быть в районе 38-40 Нм
    assert 30.0 < data["predicted_torque_nm"] < 50.0
