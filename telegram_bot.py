from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from chatbot import chatbot

async def respond(update: Update, context) -> None:
    # Получаем сообщение от пользователя
    user_message = update.message.text
    # Получаем ответ от чатбота (гигачата)
    response = chatbot(user_message)
    response = response["answer"] + f"\nКлассификатор 1го уровня: {response["class_1"]}" + f"\nКлассификатор 2го уровня: {response["class_2"]}"
    # Ответ пользователю
    await update.message.reply_text(response)

# Основная функция для запуска бота
def main():
    # Ваш токен API от BotFather
    TOKEN = "7060658208:AAH4j2I6uF-VPijn3ySSI9kstW_lAlwvvBg"

    # Создание объекта Application с токеном
    application = Application.builder().token(TOKEN).build()

    # Добавляем обработчик сообщений, который реагирует на любое текстовое сообщение
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    # Запуск бота
    application.run_polling()  # Запуск бота с поддержкой Polling

# Запуск основной функции
if __name__ == '__main__':
    main()