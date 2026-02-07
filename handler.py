import os
import sys
import runpod
from vllm import LLM, SamplingParams

print("--- [STARTUP] SCAN DU VOLUME ---")

# 1. Lister tout ce qui se trouve dans /workspace pour trouver le modele
base_path = '/workspace'
if os.path.exists(base_path):
    print(f"Contenu de {base_path} : {os.listdir(base_path)}")
else:
    print(f"ERREUR : {base_path} est introuvable. Verifiez le montage du volume.")

# 2. Definition du chemin (A ajuster selon le log de la ligne au-dessus)
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"
os.environ["HF_HUB_OFFLINE"] = "1"

llm = None

# 3. Tentative de chargement avec logs precis
try:
    if os.path.exists(MODEL_PATH):
        print(f"--- [DEBUG] Dossier trouve, lancement de vLLM ---")
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2, 
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enforce_eager=True
        )
        print("--- [SUCCESS] Modele charge ---")
    else:
        print(f"--- [ERREUR] Le chemin {MODEL_PATH} n'existe toujours pas ---")
except Exception as e:
    print(f"--- [CRASH VLLM] : {str(e)} ---")

def handler(job):
    if llm is None:
        return {"error": "Le modele n'est pas charge. Verifiez les logs de demarrage."}
    
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt", "Hello")
        sampling_params = SamplingParams(temperature=0.3, max_tokens=500)
        
        outputs = llm.generate([prompt], sampling_params)
        return {"output": outputs[0].outputs[0].text}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

