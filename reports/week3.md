## Week 3: Evaluation and Refinement

### Evaluate the quality of the generated questions using metrics.
Были написаны отдельные скрипты для оценки качества генерации вопросов для трёх моделей Llama 3 8B Fine-tune, Llama 3 8B and Llama 3 70B.

Для ознакомления с кодом прошу обратиться к следующим файлам:
**model/evaluate/evaluate_finetune_llama3_8b.py** - оценка Llama 3 8B Fine-tune
**model/evaluate/evaluate_llama3_8b.py** - оценка Llama 3 8B-Instruct
**model/evaluate/evaluate_llama3_70b.py** - оценка Llama 3 70B-Instruct

Результаты оценки трёх моделей были записаны в файл: model/evaluate/loss.txt
Результаты следующие:
Fine-tuned Llama 3 8B Model Loss: 2.8100526666641237
Llama 3 8B Model Loss: 3.3812790155410766
Llama 3 70B Model Loss: 2.616840671300888

Как видим, три эпохи обучения на данном датасете позволили Fine-tuned Llama 3 8B приблизиться к Llama 3 70B

Необходимости в контексте данного проекта исползовать другие технологии или дальнейшее обучение отсутсвует. 
Мне остаётся только запустить сервер Ollama с моделью и написать демо сайт, который позволит обратиться к моделе и самостоятельно испытать её.

Так же, необходимо сделать финальный репорт и презентацию, где будут суммированны и изложены все основные моменты и результаты работы.
