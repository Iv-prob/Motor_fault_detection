import sqlite3
import pandas as pd

# 1. Загружаем полный CSV файл (10 000 строк)
df = pd.read_csv("ai4i2020.csv")

# ПРИНУДИТЕЛЬНО И ЧЕТКО ПЕРЕИМЕНОВЫВАЕМ ТОЛЬКО НУЖНЫЕ НАМ КОЛОНКИ
# Это защитит нас от любых неожиданных названий в исходном файле
rename_dict = {
    'Air temperature [K]': 'Air_temperature',
    'Process temperature [K]': 'Process_temperature',
    'Rotational speed [rpm]': 'Rotational_speed',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'Tool_wear',
    'Machine failure': 'Machine_failure'
}
df = df.rename(columns=rename_dict)

# 2. Создаем подключение к виртуальной базе данных в ОЗУ
conn = sqlite3.connect(':memory:')

# Переносим DataFrame в таблицу factory_telemetry
df.to_sql('factory_telemetry', conn, index=False, if_exists='replace')

# 3. НАШ НАДЕЖНЫЙ SQL-ЗАПРОС
sql_query = """
SELECT 
    Machine_failure, 
    AVG(Torque) AS "Средний_Момент", 
    AVG(Rotational_speed) AS "Средняя_Скорость" 
FROM 
    factory_telemetry 
GROUP BY 
    Machine_failure;
"""

# 4. Выполняем SQL-запрос и выводим результат
result = pd.read_sql_query(sql_query, conn)

print("=== РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ВАШЕГО SQL-ЗАПРОСА ===")
print(result)

# Закрываем соединение с базой
conn.close()