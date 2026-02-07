FROM vllm/vllm-openai:v0.7.2

# Installation de runpod
RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

# Correction : On utilise python3 explicitement
ENTRYPOINT ["python3", "-u"]
CMD ["/handler.py"]
