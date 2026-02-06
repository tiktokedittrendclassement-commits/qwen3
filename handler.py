import runpod
from vllm import LLM, SamplingParams

# IMPORTANT : Le chemin pointe vers ton Network Volume
MODEL_PATH = "/models/qwen3-fp8"

# On charge le modèle (vLLM va le chercher dans le volume branché)
llm = LLM(
    model=MODEL_PATH, 
    quantization="fp8", 
    gpu_memory_utilization=0.90, # Laisse un peu de place pour le système
    enforce_eager=True # Aide à la stabilité sur RunPod
)

def handler(job):
    job_input = job['input']
    prompt = job_input.get("prompt", "")
    
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        max_tokens=job_input.get("max_tokens", 2000),
        top_p=0.95
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

runpod.serverless.start({"handler": handler})
