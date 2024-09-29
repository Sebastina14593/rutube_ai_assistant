from fastapi import FastAPI
from pydantic import BaseModel
import chatbot  # Импортируйте ваш основной скрипт

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(question: Question):
    gigachat_answer = chatbot.gigachat_response(question.question)
    answer = gigachat_answer["answer"]
    class_1 = gigachat_answer["class_1"]
    class_2 = gigachat_answer["class_2"]
    return {"answer": answer, "class_1": class_1, "class_2": class_2}