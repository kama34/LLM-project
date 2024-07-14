## Week 3: Evaluation and Development

### Evaluate the quality of the generated questions using metrics.
Separate scripts were written to evaluate the quality of the generated questions for the three models Llama 3 8B Fine-tune, Llama 3 8B and Llama 3 70B.

Please refer to the following files to familiarize yourself with the code:
**model/evaluate/evaluate_finetune_llama3_8b.py** - Llama 3 8B Fine-tune evaluation
**model/evaluate/evaluate/evaluate_llama3_8b.py** - evaluation of Llama 3 8B-Instruct
**model/evaluate/evaluate_llama3_70b.py** - evaluation of Llama 3 70B-Instruct

The evaluation results of the three models were written to the file: model/evaluate/loss.txt
The results are as follows:
- Fine-tuned Llama 3 8B Model Loss: 2.8100526666641237
- Llama 3 8B Model Loss: 3.3812790155410766
- Llama 3 70B Model Loss: 2.616840671300888

As we can see, three epochs of training on this dataset allowed Fine-tuned Llama 3 8B to approach Llama 3 70B

There is no need to use other technologies or further training in the context of this project. 
I only need to start the Ollama server with the model and write a demo site that will allow to access the model and test it myself.

Also, it is necessary to make a final report and presentation, which will summarize and outline all the main points and results of the work.