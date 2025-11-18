import requests
import json
import random
import time # Импортируем time

url = 'http://127.0.0.1:8000/esp/receive/'

for i in range(5): # Цикл из 5 итераций
    data_to_send = {
        "voltage": round(random.uniform(0.0, 12.0), 2),
        "current": round(random.uniform(0.0, 5.0), 2),
        "pwm_duty_cycle": round(random.uniform(0.0, 1.0), 3),
        "pwm_frequency": round(random.uniform(10.0, 1000.0), 1),
        "pwm_period": round(random.uniform(0.001, 0.1), 4),
        "percentage_param": round(random.uniform(0, 100), 1)
    }

    json_data = json.dumps(data_to_send)
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json_data, headers=headers)
        if response.status_code == 200:
            print(f"[{i+1}] Данные успешно отправлены: {data_to_send}")
        else:
            print(f"[{i+1}] Ошибка. Статус: {response.status_code}, Ответ: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[{i+1}] Ошибка запроса: {e}")

    time.sleep(2) # Пауза 2 секунды между запросами

print("Отправка завершена.")