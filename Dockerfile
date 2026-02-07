# Utilisation d'une version spécifique pour éviter les erreurs de driver CUDA
FROM vllm/vllm-openai:v0.7.2

LABEL version="1.2"
LABEL description="Qwen3-Coder Serverless - Fixed CUDA Compatibility"

# Installation du SDK RunPod
RUN pip install --no-cache-dir runpod

# Copie du script handler
COPY handler.py /handler.py

# Force l'utilisation de Python pour éviter que vLLM ne lance son propre serveur
ENTRYPOINT ["python", "-u"]
CMD ["/handler.py"]
