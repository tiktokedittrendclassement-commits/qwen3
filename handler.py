import runpod
import os
from vllm import LLM, SamplingParams

# On force Python à vider les logs immédiatement
os.environ["PYTHONUNBUFFERED"] = "1"

print("--- INITIALISATION DU WORKER ---")

MODEL_PATH = "/models/qwen3-fp8"

# On baisse l'utilisation à 0.85 (soit 85% de la VRAM) 
# pour laisser 3.6 Go de libre pour le système et éviter le freeze.
try:
    print(f"Chargement du modèle : {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="fp8",
        gpu_memory_utilization=0.85, 
        enforce_eager=True,
        max_model_len=4096
    )
    print("--- MODÈLE CHARGÉ AVEC SUCCÈS ---")
except Exception as e:
    print(f"ERREUR DURANT LE CHARGEMENT : {e}")

def handler(job):
    print(f"Nouveau job reçu : {job['id']}")
    job_input = job['input']
    prompt = job_input.get("prompt", "")
    
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        max_tokens=job_input.get("max_tokens", 2000),
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

runpod.serverless.start({"handler": handler})

