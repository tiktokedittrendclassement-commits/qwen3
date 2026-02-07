import runpod
from vllm import LLM, SamplingParams
import os

# Configuration du modèle
# On s'assure que vLLM cherche le modèle uniquement dans ton volume monté
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

# Initialisation du moteur LLM (C'est ici que le chargement se fait)
# tensor_parallel_size=2 pour utiliser tes deux GPUs
print(f"Chargement du modèle depuis {MODEL_PATH}...")
try:
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.90 # Laisse un peu de place pour le système
    )
except Exception as e:
    print(f"Erreur lors du chargement : {e}")
    raise e

def handler(job):
    """
    Fonction appelée à chaque requête API
    """
    try:
        job_input = job['input']
        
        # Extraction des paramètres de la requête
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Tu dois fournir un prompt."}

        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.3),
            max_tokens=job_input.get("max_tokens", 2000),
            top_p=job_input.get("top_p", 0.95),
            repetition_penalty=1.1
        )

        # Génération de la réponse
        outputs = llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text

        return {"generated_text": result}

    except Exception as e:
        return {"error": f"Erreur pendant la génération : {str(e)}"}

# Lancement du worker RunPod
print("Worker Serverless prêt !")
runpod.serverless.start({"handler": handler})
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

runpod.serverless.start({"handler": handler})


