import uvicorn
import pandas as pd
from fastapi import FastAPI
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Инициализируем веб-приложение
app = FastAPI(title="Motor Predictive Analytics API")

# 2. Обучаем нашу модель на сервере
df = pd.read_csv("ai4i2020.csv")
rename_dict = {
    'Process temperature [K]': 'Process_temp',
    'Rotational speed [rpm]': 'Rotational_speed',
    'Torque [Nm]': 'Torque'
}
df = df.rename(columns=rename_dict)

X = df[['Rotational_speed', 'Process_temp']]
y = df['Torque']

# Используем полиномиальный подход
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)


# 3. Создаём асинхронный эндпоинт
@app.get("/predict")
async def predict_torque(speed: float, temp: float):
    """
    Асинхронно принимает обороты (speed) и температуру (temp) мотора,
    делает предсказание крутящего момента и возвращает ответ в формате JSON.
    """
    # Создаем DataFrame для одной входящей точки данных
    input_data = pd.DataFrame([[speed, temp]], columns=['Rotational_speed', 'Process_temp'])

    # Искривляем признаки под полином
    input_poly = poly.transform(input_data)

    # Делаем предсказание
    prediction = model.predict(input_poly)[0]

    # Возвращаем бэкенд-ответ (JSON dict)
    return {
        "status": "success",
        "input": {
            "rotational_speed_rpm": speed,
            "process_temperature_k": temp
        },
        "predicted_torque_nm": round(float(prediction), 2)
    }


# Запускаем сервер при старте скрипта
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
