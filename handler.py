import runpod
from llama_cpp import Llama

# Chargement du modèle au démarrage du conteneur
print("Chargement du modèle Qwen 30B...")
llm = Llama(
    model_path="Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.i1-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096
)

def handler(event):
    # Récupérer le prompt envoyé par l'utilisateur
    job_input = event["input"]
    prompt = job_input.get("prompt", "Hello")
    max_tokens = job_input.get("max_tokens", 1000)

    # Génération
    output = llm(
        f"User: {prompt}\nAssistant:",
        max_tokens=max_tokens,
        stop=["User:"]
    )

    return output["choices"][0]["text"]

runpod.serverless.start({"handler": handler})