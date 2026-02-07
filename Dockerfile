FROM vllm/vllm-openai:v0.7.2

RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

# On s'assure d'utiliser python3
ENTRYPOINT ["python3", "-u"]
CMD ["/handler.py"]
