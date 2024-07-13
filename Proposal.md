**Title: Generating Open-Ended Problems from Studied Topics using Natural Language Processing**

**Author: Dmitrij Kamyshnikov**
Innopolis University
Email: d.kamyshnikov@innopolis.university

---

**Project Idea**

The project aims to develop a system that can take a specific topic as input and generate open-ended problems related to that topic. This system will utilize natural language processing techniques to create diverse and challenging questions that encourage deeper understanding and critical thinking.

**Method/Technique**

The proposed method involves leveraging the capabilities of state-of-the-art LLMs such as Ollama and LLaMA 3 70B:

1. **Topic Extraction:** Utilize the models' natural language understanding to extract key themes and concepts from input texts.
2. **Question Generation:** Employ the LLMs to generate questions based on the identified themes. These models will be fine-tuned or prompted to ensure the generation of high-quality, relevant questions.
3. **Diversity and Quality Assurance:** Implement strategies to ensure the generated questions are diverse, relevant, and of high quality. This might include prompt engineering and iterative refinement.

**Dataset Explanation and Accessible Link**

We will use three datasets for training and evaluation:

1. **Source:** Use publicly available educational texts, academic articles, and textbooks as the primary sources.
2. **Preprocessing:** Clean and preprocess the dataset to ensure it is suitable for training and fine-tuning the LLMs.
3. **Access:** Public datasets like those available on Kaggle or academic repositories will be used. For example, the SQuAD dataset can be a starting point for question-answer pairs.

**Timeline with Individual Contributions**

**Week 1: Preparation and Initial Setup**

1. **Day 1-2:** Literature review on current state-of-the-art techniques in question generation and large language models.
2. **Day 3-5:** Collect and preprocess the dataset. Ensure that the data is cleaned and formatted correctly for use with Ollama and LLaMA 3 70B.
3. **Day 6-7:** Setup the development environment, including installing necessary libraries and setting up the models.

**Week 2: Model Training and Development**

1. **Day 1-2:** Fine-tune Ollama for topic extraction using the preprocessed dataset.
2. **Day 3-5:** Train LLaMA 3 70B on the dataset to generate open-ended questions based on the extracted topics.
3. **Day 6-7:** Develop the initial prototype of the system that integrates topic extraction and question generation functionalities.

**Week 3: Evaluation and Refinement**

1. **Day 1-2:** Evaluate the quality of the generated questions using metrics like BLEU, ROUGE, and human feedback.
2. **Day 3-4:** Refine the models and improve the prompts based on the evaluation results to enhance question diversity and relevance.
3. **Day 5-6:** Conduct thorough testing and debugging of the system to ensure stability and reliability.
4. **Day 7:** Prepare the final project report and presentation, summarizing the methodology, results, and future work.

**References**

1. SciQ Dataset, https://allenai.org/data/sciq
2. SQuAD dataset, https://rajpurkar.github.io/SQuAD-explorer/
