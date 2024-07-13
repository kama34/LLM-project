import ollama from 'ollama';

const runOllamaTest = async () => {
    const response = await ollama.chat({
        model: 'llama2',
        messages: [{role: 'user', content: 'Why is the sky blue?'}],
    });
    console.log(response.message.content);
};

runOllamaTest();
