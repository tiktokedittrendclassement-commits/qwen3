import runpod
from vllm import LLM, SamplingParams
import os

# Configuration du chemin du modèle sur ton volume EU-RO-1
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

print("--- [STARTUP] Initialisation du worker Serverless ---")

# On charge le modèle une seule fois au démarrage du conteneur
try:
    print(f"--- [vLLM] Chargement du modèle depuis {MODEL_PATH} ---")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,      # Utilise tes 2 GPUs (4090/3090/A5000)
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # Laisse 15% de VRAM libre pour éviter les crashs
        max_model_len=8192            # Limite la fenêtre pour économiser de la mémoire
    )
    print("--- [SUCCESS] Modèle chargé et prêt sur les 2 GPUs ! ---")
except Exception as e:
    print(f"--- [ERROR] Échec critique du chargement : {str(e)} ---")
    raise e

def handler(job):
    """
    Cette fonction est exécutée à chaque appel API.
    """
    try:
        # 1. Extraction des données
        job_input = job.get('input', {})
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Tu as oublié le prompt dans le JSON d'entrée."}

        print(f"--- [JOB] Nouveau prompt reçu (Taille: {len(prompt)} caractères) ---")

        # 2. Configuration des paramètres de génération
        # On récupère les valeurs du JSON ou on met des valeurs par défaut
        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.2),
            max_tokens=job_input.get("max_tokens", 2000),
            top_p=job_input.get("top_p", 0.95),
            repetition_penalty=job_input.get("repetition_penalty", 1.1)
        )

        # 3. Génération avec vLLM
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        print("--- [JOB] Génération terminée avec succès ---")

        # 4. Retour du résultat
        return {
            "output": generated_text,
            "tokens_generated": len(outputs[0].outputs[0].token_ids)
        }

    except Exception as e:
        print(f"--- [ERROR] Erreur pendant la génération : {str(e)} ---")
        return {"error": str(e)}

# Lancement du SDK RunPod
runpod.serverless.start({"handler": handler})

