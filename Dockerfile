# Utilisation d'une image optimisée pour vLLM
FROM vllm/vllm-openai:latest

# Métadonnées de version
LABEL version="1.0"
LABEL maintainer="TonPseudo"
LABEL description="Qwen3-Coder Serverless V1 - vLLM"

# Installation du SDK Runpod (vLLM est déjà dans l'image de base)
RUN pip install --no-cache-dir runpod

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# Copie du handler
COPY handler.py /handler.py

# On expose le port par défaut (optionnel en serverless mais propre)
EXPOSE 8000

# Commande de lancement
CMD ["python", "-u", "/handler.py"]
