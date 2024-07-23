import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация состояния сессии
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_loading = False

def load_model():
    st.session_state.model_loading = True
    model_path = '/home/kama/project/model/finetuned_llama3'

    # Загрузка модели и токенизатора
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded.")
    st.session_state.model_loaded = True
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.model_loading = False

def generate_questions(context):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    # Пример системы запроса
    system_prompt = "Extract possible questions from the given context."
    input_text = f"Context: {context}"

    formatted_prompt = f"System: {system_prompt}\nUser: {input_text}"

    inputs = tokenizer(formatted_prompt, return_tensors='pt',
                       truncation=True, padding=True, max_length=1024)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    st.title('Question Generation using Fine-tuned Model')

    # Предопределенные контексты
    predefined_contexts = [
        "Climate change is a long-term change in the Earth's climate, especially a change due to an increase in the average atmospheric temperature.",
        "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "Quantum computing is an area of computing focused on developing computers that use quantum bits or qubits, which can exist in multiple states simultaneously."
    ]

    # Выпадающий список для выбора контекста
    selected_context = st.selectbox(
        "Choose a predefined context or enter your own:",
        [""] + predefined_contexts
    )

    # Поле для ввода пользовательского контекста
    context_input = st.text_area(
        "Or type your own context here:", value=selected_context)

    if st.button('Generate Questions'):
        if not context_input.strip():
            st.warning("Please provide a context to generate questions.")
        else:
            if st.session_state.model_loading:
                st.warning("Model is still loading. Please try again in a moment.")
            elif not st.session_state.model_loaded:
                st.warning("Model is not loaded. Please wait.")
            else:
                with st.spinner('Generating questions...'):
                    # Генерация вопросов
                    questions = generate_questions(context_input)
                    st.subheader("Generated Questions:")
                    st.write(questions)

if __name__ == "__main__":
    if not st.session_state.model_loaded and not st.session_state.model_loading:
        with st.spinner('Loading model in the background...'):
            load_model()

    main()
