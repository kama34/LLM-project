import requests


# Функция для отправки запросов к API Ollama
def call_ollama_api(prompt, model="llama2"):
    url = "http://localhost:11434/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "prompt": prompt
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


# Пример использования
response = call_ollama_api("Why is the sky blue?", model="llama3")
if response:
    print(response['choices'][0]['text'])
