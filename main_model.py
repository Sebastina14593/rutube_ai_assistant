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
    chain = prompt | model

    return(model)