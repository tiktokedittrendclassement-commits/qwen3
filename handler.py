import runpod
from vllm import LLM, SamplingParams
import os

# CONFIGURATION SÉCURISÉE
# On interdit à vLLM de tenter des téléchargements externes
os.environ["HF_HUB_OFFLINE"] = "1"
# Chemin vérifié via ta commande 'find'
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

print("--- [STARTUP] Initialisation du worker Qwen3 ---")

try:
    print(f"--- [vLLM] Chargement local depuis {MODEL_PATH} ---")
    # Initialisation avec Tensor Parallelism pour tes 2 GPUs
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,      # Obligatoire : utilise 2 GPUs
        trust_remote_code=True,
        gpu_memory_utilization=0.85, # Marge de sécurité pour la VRAM
        max_model_len=8192,          # Limite le contexte pour la stabilité
        enforce_eager=True           # Aide parfois à la compatibilité driver
    )
    print("--- [SUCCESS] Modèle chargé et prêt sur 2 GPUs ! ---")
except Exception as e:
    print(f"--- [ERROR] Échec du chargement : {str(e)} ---")
    raise e

def handler(job):
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Champ 'prompt' manquant dans 'input'."}

        # Configuration de la réponse
        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.3),
            max_tokens=job_input.get("max_tokens", 2000),
            top_p=job_input.get("top_p", 0.95),
            repetition_penalty=1.1
        )

        # Génération
        outputs = llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text

        return {"output": result}

    except Exception as e:
        return {"error": f"Erreur de génération : {str(e)}"}

# Lancement du service RunPod
runpod.serverless.start({"handler": handler})
