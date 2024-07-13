import ollama

# Function to extract topics
def extract_topics(text):
    stream = ollama.chat(
        model='llama3:70b',
        messages=[{'role': 'user', 'content': f'Extract the main topics from the following text: {text}. Write only 3 topics. '}],
        stream=False,
    )
    topics = []
    # for chunk in stream:
    #     topics.append(chunk['message']['content'])
    return stream

# Example usage
text = "Artificial Intelligence (AI) is a field of computer science that aims to create machines that can perform tasks that would normally require human intelligence. These tasks include things like visual perception, speech recognition, decision-making, and language translation."
topics = extract_topics(text)
print("Extracted Topics:", topics)
