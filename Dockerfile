# Utiliser une image CUDA stable
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Éviter les questions interactives lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installation de Python, wget, outils de build ET nano
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    nano \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation de llama-cpp-python avec support GPU (RTX 4090)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip3 install --upgrade pip
RUN pip3 install llama-cpp-python runpod

WORKDIR /app

# Téléchargement du modèle (Optionnel si tu utilises un volume réseau, 
# mais obligatoire si tu veux une image "tout-en-un")
RUN wget https://huggingface.co/mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF/resolve/main/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.i1-Q4_K_M.gguf

# Copier tes scripts
COPY handler.py .
COPY chat.py . 

# Lancer le handler serverless par défaut
CMD ["python3", "-u", "handler.py"]