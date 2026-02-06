import runpod
from vllm import LLM, SamplingParams

# Chargement du modèle au démarrage (plus rapide en FP8)
llm = LLM(model="/app/model", quantization="fp8", gpu_memory_utilization=0.90)

def handler(job):
    job_input = job['input']
    prompt = job_input.get("prompt", "Hello")
    max_tokens = job_input.get("max_tokens", 1000)
    temperature = job_input.get("temperature", 0.7)

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    # Génération
    outputs = llm.generate([prompt], sampling_params)
    
    # On renvoie le texte
    return outputs[0].outputs[0].text

runpod.serverless.start({"handler": handler})
