import runpod
from vllm import LLM, SamplingParams
import os

# Configuration environnement
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

print("--- DÉMARRAGE DU WORKER TEST ---")

try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2, # Assure-toi d'avoir séléctionné 2 GPUs sur RunPod
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True
    )
    print("--- MODÈLE CHARGÉ AVEC SUCCÈS ---")
except Exception as e:
    print(f"--- ERREUR : {str(e)} ---")
    raise e

def handler(job):
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Prompt vide"}

        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.3),
            max_tokens=job_input.get("max_tokens", 1000)
        )

        outputs = llm.generate([prompt], sampling_params)
        return {"output": outputs[0].outputs[0].text}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
