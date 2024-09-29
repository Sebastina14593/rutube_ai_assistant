# Проект: Telegram-бот для обработки запросов с использованием LLM

Этот проект представляет собой Telegram-бот, который отвечает на запросы пользователей с использованием LLM-модели для поиска ответов в базе знаний (БЗ). Бот классифицирует запросы пользователей на два уровня и предоставляет ответы на основе загруженных данных.

## Содержание
- [Функциональные возможности](#функциональные-возможности)
- [Требования](#требования)
- [Установка и настройка](#установка-и-настройка)
- [Описание структуры проекта](#описание-структуры-проекта)
- [Использование](#использование)
- [Лицензия](#лицензия)

## Функциональные возможности
1. Получение сообщений от пользователей в Telegram и автоматическая обработка.
2. Поиск ответа в базе знаний, если запрос полностью совпадает с вопросом в БЗ.
3. В случае отсутствия точного совпадения — использование LLM для генерации ответа с использованием данных БЗ.
4. Классификация запросов на два уровня: 
   - **Классификатор 1-го уровня** (Основная категория)
   - **Классификатор 2-го уровня** (Подкатегория)

## Требования
- Python 3.8 или выше
- Модели Ollama и Hugging Face для работы с LLM
- Библиотеки:
  - `telegram`
  - `dotenv`
  - `pandas`
  - `lancedb`
  - `langchain`
  - `collections`
  - `string`

## Установка и настройка

### 1. Клонирование репозитория
```bash
git clone https://github.com/Sebastina14593/rutube_ai_assistant.git
cd telegram-bot-llm
```

### 2. Установка зависимостей
Установите все необходимые библиотеки, указанные в файле requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения
Создайте файл .env в корневом каталоге и добавьте туда следующие переменные:
```bash
TELEGRAM_TOKEN = your-telegram-bot-token
```

### 4. Добавление базы знаний
Убедитесь, что файл базы знаний в формате Excel (БЗ.xlsx) размещен в папке content/. Файл должен содержать следующие колонки:
- Вопрос из БЗ
- Ответ из БЗ
- Классификатор 1 уровня
- Классификатор 2 уровня

## Описание структуры проекта
### Файл: telegram_bot_py.py

Основной файл, в котором описана логика обработки сообщений через Telegram API.
```bash
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from chatbot import chatbot

async def respond(update: Update, context) -> None:
    user_message = update.message.text
    response = chatbot(user_message)
    response_text = response["answer"] + f"\nКлассификатор 1-го уровня: {response['class_1']}" + f"\nКлассификатор 2-го уровня: {response['class_2']}"
    await update.message.reply_text(response_text)

def main():
    TOKEN = os.getenv('TELEGRAM_TOKEN')
    application = Application.builder().token(TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))
    application.run_polling()

if __name__ == '__main__':
    main()
```

## Файл: chatbot.py
Логика работы чатбота, включая загрузку базы знаний, создание эмбеддингов и взаимодействие с моделью.

```bash
from dotenv import load_dotenv
from llm_model import create_embedding, get_answer_and_metadata, specific_read_excel, database

load_dotenv()
file_path = 'content/БЗ.xlsx'

def chatbot(user_question):
    df_source = specific_read_excel(file_path, user_question)
    if df_source.shape[0] == 1:
        return {'answer': df_source["Ответ из БЗ"].values[0], 'class_1': df_source["Классификатор 1 уровня"].values[0], 'class_2': df_source["Классификатор 2 уровня"].values[0]}
    embedding = create_embedding()
    database(embedding)
    answer = get_answer_and_metadata(user_question, embedding)
    return answer
```

## Файл: main_model.py
```bash
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def ollama_model_rutube():
    template = """
    Ответь на вопрос пользователя.
    Вот контекст: {context}
    Вопрос: {question}
    Ответ:
    """
    model = OllamaLLM(model="bambucha/saiga-llama3", temperature=0.001, top_p=0.9)
    prompt = ChatPromptTemplate.from_template(template)
    return model
```

## Использование
1. Запустите проект:
```bash
python telegram_bot_py.py
```
2. После запуска, бот будет готов к обработке сообщений в Telegram.
3. Отправьте боту сообщение, и он предоставит ответ на ваш запрос на основе базы знаний и LLM-модели.