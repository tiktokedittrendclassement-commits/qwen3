import os
import sys

# DEBUG : Affiche les infos système dès le début
print(f"--- DEBUG SYSTÈME ---")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in root: {os.listdir('/')}")

try:
    import runpod
    from vllm import LLM, SamplingParams
    print("--- Imports réussis ---")
except ImportError as e:
    print(f"--- ERREUR IMPORT : {str(e)} ---")
    sys.exit(1)

# Configuration vLLM
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

try:
    print(f"--- Tentative de chargement du modèle : {MODEL_PATH} ---")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096, # On baisse un peu pour le test
        enforce_eager=True
    )
    print("--- MODÈLE CHARGÉ AVEC SUCCÈS ---")
except Exception as e:
    print(f"--- ERREUR CHARGEMENT MODÈLE : {str(e)} ---")
    # On ne fait pas de raise ici pour laisser le worker afficher l'erreur
    sys.exit(1)

def handler(job):
    # Ton code de handler habituel
    return {"output": "Test réussi"}

runpod.serverless.start({"handler": handler})
