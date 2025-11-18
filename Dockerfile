# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код проекта в контейнер
COPY . .

# Создаем суперпользователя (опционально, если нужно)
# RUN python manage.py createsuperuser --noinput

# Команда, которая выполнится при запуске контейнера
# Запускаем Django-сервер, слушая на 0.0.0.0 (доступен снаружи контейнера)
# Порт 8000 будет мапиться на удаленной машине
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]