## Week 2: Model Training and Development

### Fine-tune Ollama for Topic Extraction
After studying the problem more carefully, studying scientific articles and thinking over the algorithm of actions, I came to the conclusion that there is no need to train the model for topic extraction from text. We can immediately make Fine-tune for generating open questions using the SQuAD dataset, as it fits us best and has enough data.


#### Train LLaMA 3 70B and 8B on the Dataset to Generate Open-Ended Questions
On the available computing resources it was not possible to make Fine-tune LLaMA 3 70B, there was not enough memory. 
However, we managed to make Fine-tune LLaMA 3 8B-Instruct. 

I won't insert the code into the report because it is very long.
To familiarize yourself with the code in detail, you should look at the following files:
model/finetune_llama3_8b.py - Fine-tune model Llama 3 8B
model/finetune_llama3_70b.py - Fine-tune attempt of the Llama 3 70B model.

Checkpoints saved during training are in model/llama3_results/
The saved model is in model/finetuned_llama3/.

It should be noted that the training took place for 3 epochs, but the calculation was that there will be 100 of them. 
Therefore, the program had to be interrupted earlier and in order to load checkpoints and then save the model, the code from the file: model/load_model_from_checkpoint.py was written and executed.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = './llama3_results/checkpoint-7000'.

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.save_pretrained('./finetuned_llama3')
tokenizer.save_pretrained('./finetuned_llama3')
```

You can use the following commands from the root of the repository to run the models
```bash
python model/evaluate/evaluate_finetune_llama3_8b.py 
```

```bash
python model/evaluate/evaluate_llama3_8b.py 
```

```bash
python model/evaluate/evaluate_llama3_70b.py 
```

The next step will be to evaluate the resulting models and deploy this model on the university server, possibly using Ollama and Streamlit