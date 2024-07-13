from ollama import Ollama

# Создание экземпляра Ollama
ollama = Ollama()

# Функция для отправки запросов к API Ollama
def call_ollama_api(prompt, model="llama2"):
    response = ollama.complete(prompt, model=model)
    return response

# Пример использования
response = call_ollama_api("Why is the sky blue?", model="llama3")
if response:
    print(response['choices'][0]['text'])
