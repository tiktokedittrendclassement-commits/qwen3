import runpod
from vllm import LLM, SamplingParams

# Chemin vers ton volume monté
MODEL_PATH = "/workspace/Qwen3-Coder-FP8"

print(f"Chargement du modèle vLLM depuis {MODEL_PATH}...")
try:
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=2, # Utilise tes 2 GPUs
        trust_remote_code=True,
        gpu_memory_utilization=0.90
    )
except Exception as e:
    print(f"Erreur de chargement : {e}")
    raise e

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get("prompt")
        
        if not prompt:
            return {"error": "Prompt manquant"}

        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.3),
            max_tokens=job_input.get("max_tokens", 2000)
        )

        outputs = llm.generate([prompt], sampling_params)
        return {"generated_text": outputs[0].outputs[0].text}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
