cat <<EOF > /workspace/chat.py
from llama_cpp import Llama

llm = Llama(
    model_path="/workspace/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.i1-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False
)

print("\n--- IA MODE CODEUR (NE COUPE PLUS) ! ---")

while True:
    user_input = input("\nToi: ")
    if user_input.lower() == 'exit':
        break
    
    output = llm(
        f"User: {user_input}\nAssistant:",
        max_tokens=3000,
        temperature=0.1,
        stop=["User:", "Assistant:"], 
        echo=False
    )
    
    print(f"\nIA: {output['choices'][0]['text']}")
EOF