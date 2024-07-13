import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'\W', ' ', text)  # Удаление небуквенных символов
    text = text.lower()  # Приведение текста к нижнему регистру
    tokens = word_tokenize(text)  # Токенизация текста
    return ' '.join(tokens)


# Пример использования для SQuAD
import json

with open('train-v2.0.json') as f:
    squad_data = json.load(f)

texts = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        text = paragraph['context']
        cleaned_text = preprocess_text(text)
        texts.append(cleaned_text)

print(texts[:5])  # Вывод первых 5 очищенных текстов
