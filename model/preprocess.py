import json
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


def preprocess_squad(file_path, output_path):
    with open(file_path) as f:
        squad_data = json.load(f)

    processed_data = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            cleaned_context = preprocess_text(context)
            for qa in paragraph['qas']:
                question = qa['question']
                cleaned_question = preprocess_text(question)
                processed_data.append({
                    'context': cleaned_context,
                    'question': cleaned_question
                })

    print(processed_data[:5])  # Вывод первых 5 очищенных текстов

    with open(output_path, 'w') as f:
        json.dump(processed_data, f)


def preprocess_sciq(file_path, output_path):
    with open(file_path) as f:
        sciq_data = json.load(f)

    processed_data = []
    for entry in sciq_data:
        support = entry['support']
        question = entry['question']
        cleaned_support = preprocess_text(support)
        cleaned_question = preprocess_text(question)
        processed_data.append({
            'support': cleaned_support,
            'question': cleaned_question
        })

    print(processed_data[:5])  # Вывод первых 5 очищенных текстов
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)


def main():
    print("\nSQuAD preprocessing train\n")
    preprocess_squad('/home/kama/project/data/SQuAD/train-v2.0.json',
                     '/home/kama/project/data/processed_squad_train.json')
    print("\nSQuAD preprocessing dev\n")
    preprocess_squad('/home/kama/project/data/SQuAD/dev-v2.0.json',
                     '/home/kama/project/data/processed_squad_dev.json')

    print("\nSciQ preprocessing train\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json',
                    '/home/kama/project/data/processed_sciq_train.json')
    print("\nSciQ preprocessing test\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json',
                    '/home/kama/project/data/processed_sciq_test.json')
    print("\nSciQ preprocessing valid\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/valid.json',
                    '/home/kama/project/data/processed_sciq_valid.json')


if __name__ == "__main__":
    main()
