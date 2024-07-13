import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch

print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
print(f"torch.cuda.current_device = {torch.cuda.current_device()}")

# Load dataset
import json

with open('/home/kama/project/data/SQuAD/train-v2.0.json', 'r') as f:
    data = json.load(f)

# print(data['data'][0])

for q in data['data']:
    print(q)
    q['context'] = str(q['paragraphs']['context'])
    q['question'] = str(q['paragraphs']['qas']['question'])
    q['answers'] = str(q['paragraphs'][''])
    break

data[0]

df = pd.DataFrame(data)
df.head()

# Setup dataset
MODEL_NAME = "meta-llama/Llama-3-70b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token, tokenizer.eos_token

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)


def gen_batches_train():
    for sample in iter(ds):
        # Extract instruction and input from the sample
        system_prompt = "Ты профессиональный экзаменатор с глубоким знанием предмета. Твоя задача - помощь в составлении вопросов для студентческого экзамена."
        input_text = f"# Вопрос: {sample['question']}\n# Правильный ответ: {sample['correct_answer']}\n\nСоздай 3 правдоподобных, но неправильных ответа (дистракторов) для данного вопроса. Cгенерируй 3 неправильных ответа (дистрактора) в следующем формате:\n# Дистракторы:\n - <неправильный ответ 1>\n - <неправильный ответ 2>\n - <неправильный ответ 3>.\nНе добавляй номера или буквы к ответам."
        out_text = f"# Дистракторы:\n - {sample['distractor_1']}\n - {sample['distractor_2']}\n - {sample['distractor_3']}"
        formatted_prompt = None

        formatted_prompt = tokenizer.apply_chat_template([{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": input_text
        }, {
            "role": "assistant",
            "content": out_text
        }], tokenize=False, add_generation_prompt=False) + '<|end_of_text|>'

        yield {'text': formatted_prompt}


next(gen_batches_train())

# Prepare model
device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

tokenizer.pad_token = tokenizer.eos_token

# Training
training_arguments = TrainingArguments(
    output_dir='./saiga_results',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=5,
    learning_rate=3e-4,
    fp16=False,
    bf16=True,
    num_train_epochs=100,
    report_to="none"
)

train_gen = Dataset.from_generator(gen_batches_train)
tokenizer.padding_side = "right"

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

peft_model_id="./finetuned_llama3"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)