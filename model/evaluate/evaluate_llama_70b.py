import sys
import importlib.util
import json
import torch
import pandas as pd

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig


def import_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# import method
writer = import_from_path('writer', '/home/kama/project/utls/writer.py')

write_to_file = writer.write_to_file

# Load dataset
with open('/home/kama/project/data/SQuAD/dev-v2.0.json', 'r') as f:
    data = json.load(f)

parsed_data = []
for d in data['data']:
    for paragraph in d['paragraphs']:
        context = paragraph['context']
        for qas in paragraph['qas']:
            question = qas['question']
            parsed_data.append({'context': context, 'question': question})

df = pd.DataFrame(parsed_data)
ds = Dataset.from_pandas(df)


# Function to compute the loss with prompt
def compute_loss_with_prompt(model, tokenizer, dataset):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for sample in dataset:
            system_prompt = "Extract possible questions from the given context."
            input_text = f"Context: {sample['context']}"

            formatted_prompt = tokenizer.apply_chat_template([{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": input_text
            }], tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            total_count += 1
    return total_loss / total_count


# Load original models
llama_70b_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct', device_map="auto",
                                                       torch_dtype=torch.bfloat16, )

llama_70b_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct', device_map="auto",
                                                    torch_dtype=torch.bfloat16, )

# Select a subset for evaluation
eval_dataset = ds.select(range(100))

# Compute loss for each model
llama_70b_loss = compute_loss_with_prompt(llama_70b_model, llama_70b_tokenizer, eval_dataset)

write_to_file("/home/kama/project/model/evaluate/loss.txt", f"Llama 3 70B Model Loss: {llama_70b_loss}", False)

print(f"Llama 3 70B Model Loss: {llama_70b_loss}")
