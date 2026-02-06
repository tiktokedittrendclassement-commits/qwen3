FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git wget && rm -rf /var/lib/apt/lists/*

# Installation de vLLM (le moteur ultra-rapide pour FP8)
RUN pip3 install --upgrade pip
RUN pip3 install vllm runpod huggingface_hub

WORKDIR /app

# Téléchargement du modèle FP8 depuis Hugging Face
# On utilise huggingface-cli pour télécharger tout le dossier
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8', local_dir='/app/model')"

COPY handler.py .

# On lance le handler
CMD ["python3", "-u", "handler.py"]
