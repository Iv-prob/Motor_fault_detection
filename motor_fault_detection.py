import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Генерируем реалистичные данные с наложением и шумом
np.random.seed(42)
n_samples = 1500

# Здоровый мотор (но с нагрузками: температура иногда доходит до высокого уровня)
vibration_normal = np.random.normal(loc=1.8, scale=0.6, size=1200)
temp_normal = np.random.normal(loc=50.0, scale=8.0, size=1200)
noise_normal = np.random.normal(loc=62.0, scale=4.0, size=1200)
labels_normal = np.zeros(1200)

# Аномальный мотор (поломка на ранней стадии: признаки едва заметны и пересекаются с нормой)
vibration_anomaly = np.random.normal(loc=2.8, scale=0.8, size=300) # Пересекается с нормой!
temp_anomaly = np.random.normal(loc=65.0, scale=10.0, size=300)
noise_anomaly = np.random.normal(loc=68.0, scale=5.0, size=300)
labels_anomaly = np.ones(300)

# Объединяем
df = pd.DataFrame({
    'vibration_amplitude': np.concatenate([vibration_normal, vibration_anomaly]),
    'temperature': np.concatenate([temp_normal, temp_anomaly]),
    'noise_level': np.concatenate([noise_normal, noise_anomaly]),
    'fault': np.concatenate([labels_normal, labels_anomaly])
})

# Добавляем грязь: Имитируем разовые технические сбои датчиков (выбросы)
# У 2% абсолютно здоровых моторов датчик вибрации покажет всплеск в результате сбоя
noise_indices = df[df['fault'] == 0].sample(frac=0.02, random_state=42).index
df.loc[noise_indices, 'vibration_amplitude'] = df.loc[noise_indices, 'vibration_amplitude'] * 5

# Перемешиваем данные
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Пример данных с датчиков робота ---")
print(df.head(), "\n")

# 2. Подготовка к обучению
X = df[['vibration_amplitude', 'temperature', 'noise_level']]
y = df['fault']

# Делим на тренировочную и тестовую выборки (80% на 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение модели (Ансамбль деревьев решений)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Получаем вероятности и применяем кастомный порог
# predict_proba возвращает вероятности для каждого класса: [вероятность_0, вероятность_1]
y_scores = model.predict_proba(X_test)[:, 1]

# Если модель уверена в поломке хотя бы на 30%, мы классифицируем это как Поломку (1)
custom_threshold = 0.3
y_pred_custom = (y_scores >= custom_threshold).astype(int)

print(f"--- Метрики качества при кастомном пороге ({custom_threshold}) ---")
print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred_custom):.4f}")
print("\nДетальный отчет (Classification Report):")
print(classification_report(y_test, y_pred_custom, target_names=['Норма', 'Поломка']))

# 5. Важность признаков (Feature Importance)
print("\n--- Важность признаков для модели ---")
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"Важность признака '{feature}': {importance:.4f}")