import sys
import os

# --- BLOC DE DIAGNOSTIC PRIORITAIRE ---
print("--- [DEBUG] DEMARRAGE DU SCRIPT ---")
try:
    import runpod
    print("--- [DEBUG] SDK RunPod importé ---")
    import vllm
    print(f"--- [DEBUG] vLLM version {vllm.__version__} importé ---")
except Exception as e:
    print(f"--- [ERREUR CRITIQUE] Echec des imports : {str(e)} ---")
    sys.exit(1)

from vllm import LLM, SamplingParams

# Configuration
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"
os.environ["HF_HUB_OFFLINE"] = "1"

print(f"--- [DEBUG] Verification du dossier modele: {os.path.exists(MODEL_PATH)} ---")

# Chargement du modèle
llm = None
try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2, # Assure-toi d'avoir 2 GPUs séléctionnés sur RunPod
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        enforce_eager=True
    )
    print("--- [DEBUG] LLM chargé avec succès ---")
except Exception as e:
    print(f"--- [ERREUR CRITIQUE] vLLM n'a pas pu démarrer : {str(e)} ---")
    sys.exit(1)

def handler(job):
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt", "Hello")
        
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=500
        )
        
        outputs = llm.generate([prompt], sampling_params)
        return {"output": outputs[0].outputs[0].text}
    except Exception as e:
        return {"error": str(e)}

# Demarrage du service
runpod.serverless.start({"handler": handler})
