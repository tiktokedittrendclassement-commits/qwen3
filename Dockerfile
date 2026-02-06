# On utilise l'image officielle vLLM déjà optimisée
FROM vllm/vllm-openai:latest

# On installe juste runpod pour faire le lien avec ton SaaS
RUN pip install runpod

WORKDIR /app

# On copie ton handler qui va piloter l'IA
COPY handler.py .

# On lance le service
CMD ["python3", "-u", "handler.py"]
