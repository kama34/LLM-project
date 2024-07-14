import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["WANDB_DISABLED"] = "true"
import sys
sys.path.append('/home/kama/project')

from utls.writer import write_to_file

import json
import torch
from datasets import Dataset, load_metric
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig

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


# Function to compute the loss
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
            labels = inputs.input_ids.clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            total_count += 1
    return total_loss / total_count


# Load fine-tuned model
fine_tuned_model_path = './finetuned_llama3'
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Возможный вариант загрузки fine tune model
# from peft import PeftModel
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",torch_dtype=torch.bfloat16)
#
# model = PeftModel.from_pretrained(model, model_id=peft_model_id, config=peft_config)
#
# model = model.merge_and_unload()

# Load original models
llama_70b_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct', device_map="auto",
                                                       torch_dtype=torch.bfloat16, )
llama_8b_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map="auto",
                                                      torch_dtype=torch.bfloat16)

llama_70b_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct', device_map="auto",
                                                    torch_dtype=torch.bfloat16, )
llama_8b_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map="auto",
                                                   torch_dtype=torch.bfloat16, )

# Select a subset for evaluation
eval_dataset = ds.select(range(10))

# Compute loss for each model
fine_tuned_loss = compute_loss_with_prompt(fine_tuned_model, fine_tuned_tokenizer, eval_dataset)
# llama_70b_loss = compute_loss_with_prompt(llama_70b_model, llama_70b_tokenizer, eval_dataset)
# llama_8b_loss = compute_loss_with_prompt(llama_8b_model, llama_8b_tokenizer, eval_dataset)

write_to_file("./loss", fine_tuned_loss, True)

print(f"Fine-tuned Model Loss: {fine_tuned_loss}")