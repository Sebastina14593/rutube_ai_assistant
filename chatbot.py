from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document  # Импортируем класс Document для создания документов
import lancedb
import pandas as pd
from main_model import ollama_model_rutube
from collections import Counter
import string

# Загрузка переменных окружения из файла .env
load_dotenv()
AUTH = os.getenv('AUTH')

file_path = 'content/БЗ.xlsx'
content_columns = ['Вопрос из БЗ', 'Ответ из БЗ']
metadata_column =  ['Классификатор 1 уровня', 'Классификатор 2 уровня']  # Столбец, который будет в метаданных

def specific_read_excel(file_path, question_client):
    '''
    :param file_path: путь, по которому расположен файл excel с БЗ
    :param question_client: вопрос клиента
    :return: df_source: отфильрованный на точное совпадение датафрейм (либо однака строка / либо нет)
    Данная функция фильрует датафрейм на точное совпадение вопроса пользователя и вопроса представленного
    в БЗ. И если таковой имеется, то данный датафрейм будет состоять из одной строки,
    т.е. точного соовтетсвия вопросу пользователя
    '''

    # Избавляемся от всех знаков препинаний, а также приводим к нижнему регситру
    # ответ пользователя и вопросы представленные в БЗ
    punctuations = string.punctuation
    question_client = question_client.lower().replace(" ", "").translate(str.maketrans('', '', punctuations))
    # Считываем файл
    df_source = pd.read_excel(file_path)
    df_source["Вопрос из БЗ"] = df_source["Вопрос из БЗ"].apply(lambda x: x.lower().replace(" ", "").translate(str.maketrans('', '', punctuations)))
    # Фильтруем датафрейм
    df_source = df_source[df_source["Вопрос из БЗ"] == question_client]
    return df_source

# Функция для построчной загрузки данных из .xlsx файла
def load_docs_from_xlsx(file_path, content_columns, metadata_column):
    '''
    :param file_path: путь к файлу с БЗ
    :param content_columns: колонки с содержимым
    :param metadata_column: колонки с метадатой
    :return:
    '''
    # Считываем Excel файл в DataFrame
    df = pd.read_excel(file_path)

    docs = []

    # Проходимся по каждой строке DataFrame
    for _, row in df.iterrows():
        # Извлекаем содержимое для документа (например, несколько столбцов)
        content = "\n".join([f"{col}: {row[col]}" for col in content_columns])

        # Извлекаем метаданные (один столбец)
        metadata = {metadata_column[0]: row[metadata_column[0]], metadata_column[1]: row[metadata_column[1]]}

        # Создаем документ с контентом и метаданными
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        docs.append(doc)

    return docs

# Функция для получения ответа и списка метаданных
def get_answer_and_metadata(question, embedding):
    url = 'data/aidata'
    db = lancedb.connect(url)
    vector_store = LanceDB(db, embedding, 'data/aidata')
    # Создаем извлекающий компонент
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Создаем цепочку документов и цепочку извлечения
    document_chain = create_document_chain()
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

def create_embedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cpu"):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': False})

def create_vector_store(docs, embedding):
    return LanceDB.from_documents(docs, embedding=embedding)

def create_document_chain():
    llm = ollama_model_rutube()
    prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. Используй при этом только информацию из контекста. Если в контексте нет информации для ответа, сообщи об этом пользователю. Контекст: {context} Вопрос: {input} Ответ:''')
    return create_stuff_documents_chain(llm=llm, prompt=prompt)

def database(embedding):
    docs = load_docs_from_xlsx(file_path, content_columns, metadata_column)
    url = 'data/aidata'
    db = lancedb.connect(url)
    # Используем LanceDB для хранения документов с метаданными
    LanceDB.from_documents(docs, embedding=embedding, connection=db)

# Функция для получения ответа и списка метаданных
def get_answer_and_metadata(question, embedding, llm):
    url = 'data/aidata'
    db = lancedb.connect(url)
    vector_store = LanceDB(db, embedding, 'data/aidata')
    # Создаем извлекающий компонент
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Создаем цепочку документов и цепочку извлечения
    document_chain = create_document_chain(AUTH)
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

    # Выполняем запрос к цепочке и получаем отобранные документы
    results = embedding_retriever.get_relevant_documents(question)

    scores = vector_store.similarity_search_with_score(question)
    if scores[0][1] > 7.5:
        return {
            'answer': 'Вопрос распознан как нерелевантный для базы знаний RUTUBE. Пожалуйста, cформулируйте запрос точннее.',
            'class_1': 'ОТСУТСТВУЕТ', 'class_2': 'Отсутствует'}

    else:
        # Список для хранения метаданных
        # metadata_list = [doc.metadata[metadata_column] for doc in results if metadata_column in doc.metadata]
        metadata_list = [doc.metadata for doc in results]

        # Подсчет количества классификаторов первого уровня
        counter_first_level = Counter([d['Классификатор 1 уровня'] for d in metadata_list])

        # Подсчет количества классификаторов второго уровня
        counter_second_level = Counter([d['Классификатор 2 уровня'] for d in metadata_list])

        # Вызываем саму цепочку для получения ответа
        resp = retrieval_chain.invoke({'input': question})

        return {'answer': resp["answer"], 'class_1': list(counter_first_level.keys())[0],
                'class_2': list(counter_second_level.keys())[0]}

def chatbot(user_question):
    # Делаем проверку на полное совпадение вопроса
    df_source = specific_read_excel(file_path, user_question)
    if df_source.shape[0] == 1:
        return {'answer': df_source["Ответ из БЗ"].values[0], 'class_1': df_source["Классификатор 1 уровня"].values[0], 'class_2': df_source["Классификатор 2 уровня"].values[0]}
    # Создаем эмбеддинги
    embedding = create_embedding()
    # Создаем БЗ
    database(embedding)
    # Получаем ответ и метаданные (классификаторы 1 и 2)
    answer = get_answer_and_metadata(user_question, embedding)
    return answer