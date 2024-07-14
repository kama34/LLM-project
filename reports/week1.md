## **1. Preparation and Initial Setup (Week 1)**

### **Day 1-2: Literature Review**

Because a literature review requires research articles and documentation, you may use the following resources to find relevant articles and studies:

1. **Google Scholar** (https://scholar.google.com)
2. **ResearchGate** (https://www.researchgate.net)
3. **arXiv** (https://arxiv.org)
4. **IEEE Xplore** (https://ieeexplore.ieee.org)

### **Day 3-5: Data Collection and Pre-processing**

**Step 1: Search and download educational datasets**

1. **SQuAD (Stanford Question Answering Dataset)**

   - **Description:** Dataset for question-answering tasks, contains excerpts from Wikipedia and related questions.
   - **Link:** https://rajpurkar.github.io/SQuAD-explorer/
   - **Code to download:**

   ```python
   !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
   !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
   ```

2. **SciQ (Science Questions)**

   - **Description:** Contains science questions and answers that have been collected from educational texts.
   - **Link:** https://allenai.org/data/sciq
   - **Download code:**

   ```python
   !wget https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip
   !unzip sciQ.zip -d sciq_data
   ```
   
### File description

#### SQuAD
1. **/home/kama/project/data/SQuAD/dev-v2.0.json**
   - This is a test dataset file from the SQuAD dataset. It is used for model validation.

2. **/home/kama/project/data/SQuAD/train-v2.0.json**
   - The main file with the training dataset from the SQuAD dataset. It is used for model training.

3. **/home/kama/project/data/SQuAD/train-v2.0.json.1**
   - This is a duplicate of the train-v2.0.json file. It can be ignored as it contains the same data.

#### SciQ
1. **/home/kama/project/data/SciQ/dev-v2.0.json**
   - Test dataset from the SciQ dataset. Used for model validation.

2. **/home/kama/project/data/SciQ/train-v2.0.json**
   - Main file with training dataset from SciQ dataset. It is used for model training.

3. **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/license.txt**
   - License agreement for the SciQ dataset. Not required for data analysis and processing.

4. **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/readme.txt**
   - README file with a description of the dataset. Useful for understanding the data structure.

5. **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json**
   - Test dataset from the SciQ dataset. Used for model validation.

6. **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json**
   - Main file with the training dataset from the SciQ dataset. It is used for model training.

7. **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/valid.json**
   - Validation dataset from the SciQ dataset. Used for model evaluation during training.

### Files of interest for processing
For model training and testing, we will be interested in the following files:
- **/home/kama/project/data/SQuAD/train-v2.0.json**
- **/home/kama/project/data/SQuAD/dev-v2.0.json**
- **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json** **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json**
- **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json** **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json**
- **/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/valid.json****

### File structures of interest for processing
/home/kama/project/data/SQuAD/train-v2.0.json

**Structure JSON:**
```json
version
data
  title
  paragraphs
    qas
      question
      id
      answers
        text
        answer_start
      is_impossible
    context
```

**Example JSON:**
```json
version: v2.0
data: [
  title: Beyoncé
  paragraphs: [
    qas: [
      question: When did Beyonce start becoming popular?
      id: 56be85543aeaaa14008c9063
      answers: [
        text: in the late 1990s
        answer_start: 269
      ]
      is_impossible: False
    ]
    context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in H
ouston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by
 her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
  ]
]
```

/home/kama/project/data/SQuAD/dev-v2.0.json

**Structure JSON:**
```json
version
data
  title
  paragraphs
    qas
      question
      id
      answers
        text
        answer_start
      is_impossible
    context
```

/home/kama/project/data/SQuAD/dev-v2.0.json

**Structure JSON:**
```json
version
data
  title
  paragraphs
    qas
      question
      id
      answers
        text
        answer_start
      is_impossible
    context
```

**Example JSON:**
```json
version: v2.0
data: [
  title: Normans
  paragraphs: [
    qas: [
      question: In what country is Normandy located?
      id: 56ddde6b9a695914005b9628
      answers: [
        text: France
        answer_start: 159
      ]
      is_impossible: False
    ]
    context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. 
They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King C
harles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. 
Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
  ]
]
```

/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json

**Structure JSON:**
```json
question
distractor3
distractor1
distractor2
correct_answer
support
```

**Example JSON:**
```json
question: What type of organism is commonly used in preparation of foods such as cheese and yogurt?
distractor3: viruses
distractor1: protozoa
distractor2: gymnosperms
correct_answer: mesophilic organisms
support: Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the normal human body temperature. Mesophilic organisms have important uses in food preparation, including cheese, yogurt, beer and wine.
```


/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json

**Structure JSON:**
```json
question
distractor3
distractor1
distractor2
correct_answer
support
```

**Example JSON:**
```json
question: Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?
distractor3: residues
distractor1: antioxidants
distractor2: Oxygen
correct_answer: oxidants
support: Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are called oxidants (or oxidizing agents) because they can oxidize other compo
unds. In the process of accepting electrons, an oxidant is reduced. Compounds that are capable of donating electrons, such as sodium metal or cyclohexane (C6H12), are calledreducta
nts (or reducing agents) because they can cause the reduction of another compound. In the process of donating electrons, a reductant is oxidized. These relationships are summarized in Equation 3.30: Equation 3.30 Saylor URL: http://www. saylor. org/books.
```

/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/valid.json

**Structure JSON:**
```json
question
distractor3
distractor1
distractor2
correct_answer
support
```

**Example JSON:**
```json
question: Who proposed the theory of evolution by natural selection?
distractor3: Scopes
distractor1: Linnaeus
distractor2: shaw
correct_answer: darwin
support:
```


**Step 2: Data preprocessing**

Code to clean and tokenize the data:

```python
import json
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'\W', ' ', text)  # Удаление небуквенных символов
    text = text.lower()  # Приведение текста к нижнему регистру
    tokens = word_tokenize(text)  # Токенизация текста
    return ' '.join(tokens)


def preprocess_squad(file_path, output_path):
    with open(file_path) as f:
        squad_data = json.load(f)

    processed_data = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            cleaned_context = preprocess_text(context)
            for qa in paragraph['qas']:
                question = qa['question']
                cleaned_question = preprocess_text(question)
                processed_data.append({
                    'context': cleaned_context,
                    'question': cleaned_question
                })

    print(processed_data[:5])  # Output of the first 5 cleared texts

    with open(output_path, 'w') as f:
        json.dump(processed_data, f)


def preprocess_sciq(file_path, output_path):
    with open(file_path) as f:
        sciq_data = json.load(f)

    processed_data = []
    for entry in sciq_data:
        support = entry['support']
        question = entry['question']
        cleaned_support = preprocess_text(support)
        cleaned_question = preprocess_text(question)
        processed_data.append({
            'support': cleaned_support,
            'question': cleaned_question
        })

    print(processed_data[:5])  # Output of the first 5 cleared texts
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)


def main():
    print("\nSQuAD preprocessing train\n")
    preprocess_squad('/home/kama/project/data/SQuAD/train-v2.0.json',
                     '/home/kama/project/data/processed_squad_train.json')
    print("\nSQuAD preprocessing dev\n")
    preprocess_squad('/home/kama/project/data/SQuAD/dev-v2.0.json',
                     '/home/kama/project/data/processed_squad_dev.json')

    print("\nSciQ preprocessing train\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/train.json',
                    '/home/kama/project/data/processed_sciq_train.json')
    print("\nSciQ preprocessing test\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/test.json',
                    '/home/kama/project/data/processed_sciq_test.json')
    print("\nSciQ preprocessing valid\n")
    preprocess_sciq('/home/kama/project/data/SciQ/sciq_data/SciQ dataset-2 3/valid.json',
                    '/home/kama/project/data/processed_sciq_valid.json')


if __name__ == "__main__":
    main()
```

### **Day 6-7: Setting up the environment**

**Step 1: Install Libraries**

To work with the Ollama and LLaMA 3 70B models, you must install the necessary libraries and load the models.

1. **Installing the libraries**

```bash
conda install numpy pandas scikit-learn nltk
```

```bash
conda install requests json
```

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

```bash
pip install transformers
```

```bash
conda install jupyter
```

```bash
conda install matplotlib seaborn
```

```bash
conda install -c huggingface datasets
```

```bash
jupyter notebook
```

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
pip install ollama
```

```bash
pip install peft
```

```bash
pip install trl
```

```bash
ollama pull llama3:70b
```

Starting Ollama 
```bash
ollama serve
```

**Step 2: Verify that everything works**

Code to run the llama 3 70B model using requests

```python
import requests


# Function to send requests to the Ollama API
def call_ollama_api(prompt, model="llama2"):
    url = "http://localhost:11434/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt
    }

    response = requests.post(url, json=data, headers=headers)
    If response.status_code == 200:
        return response.json()
    else:
        print(f "Error: {response.status_code}")
        print(response.text)
        return None


# Example usage
response = call_ollama_api("Why is the sky blue?", model="llama3")
if response:
    print(response['choices'][0]['text'])
```

Run the code from the root of the project
```bash
python tests/ollama_test.py 
```

Code to run the llama 3 70B model using the ollama library and stream approach

```python
import ollama

stream = ollama.chat(
    model='llama3:70b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

Running the code from the root of the project
```bash
python tests/ollama_stream_test.py  
```