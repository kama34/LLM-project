from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = './llama3_results/checkpoint-6800'

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.save_pretrained('./finetuned_llama3')
tokenizer.save_pretrained('./finetuned_llama3')
