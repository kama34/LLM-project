## Week 2: Model Training and Development

### Fine-tune Ollama for Topic Extraction
Изучив задачу внимательнее, изучив научные статьи и продумав алгоритм действий, пришёл к выводу, что нет необходимости обучать модель на извлечение темы из текста. Можно сразу сделать Fine-tune на генерацию открытых вопросов, используя датасет SQuAD, так как он лучше всего нам подходит и в нём достаточно данных.


#### Train LLaMA 3 70B и 8B on the Dataset to Generate Open-Ended Questions
На имеющихся вычислительных ресурсах не удалось сделать Fine-tune LLaMA 3 70B, не хватило памяти. 
Однако удалось сделать Fine-tune LlaMA 3 8B-Instruct. 

Не буду вставлять код в отчёт, ибо он очень длинный.
Для детального ознакомления с кодом нужно рассмотреть следующие файлы:
model/finetune_llama3_8b.py - Fine-tune модели Llama 3 8B
model/finetune_llama3_70b.py - Попытка Fine-tune модели Llama 3 70B

Сохранённые во время обучения чекпоинты лежат в model/llama3_results/
Сохранённая модель лежит в model/finetuned_llama3/

Стоит уточнить, что обучени произошло на 3 эпохи, но расчёт был, что будет их 100. 
Поэтому программу пришлось прервать раньше и чтобы загрузить чекпоинты и затем выполнить сохранение модели был написан и запущен код из файла: model/load_model_from_checkpoint.py
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = './llama3_results/checkpoint-7000'

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.save_pretrained('./finetuned_llama3')
tokenizer.save_pretrained('./finetuned_llama3')
```

Для запуска моделей можно использовать следующие команды из корня репозитория
```bash
python model/evaluate/evaluate_finetune_llama3_8b.py 
```

```bash
python model/evaluate/evaluate_llama3_8b.py 
```

```bash
python model/evaluate/evaluate_llama3_70b.py 
```

В качестве дальнейшего этапа будет оценка получившихся моделей и развёртывание данной модели на сервере университета возможно с использованием Ollama и Streamlit