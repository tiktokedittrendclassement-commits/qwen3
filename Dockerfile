FROM vllm/vllm-openai:latest

# Métadonnées
LABEL version="1.1"

RUN pip install --no-cache-dir runpod

# On force l'ENTRYPOINT à python pour qu'il ignore la commande par défaut de vLLM
ENTRYPOINT ["python", "-u"]

# On lui dit quel fichier lancer
CMD ["/handler.py"]

COPY handler.py /handler.py
