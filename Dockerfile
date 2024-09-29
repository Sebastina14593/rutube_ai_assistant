# Используем официальный образ Python в качестве базового
FROM python:3.11-slim

# Устанавливаем рабочую директорию
RUN mkdir code
WORKDIR code

ADD . /code/

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Указываем команду для запуска API сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]