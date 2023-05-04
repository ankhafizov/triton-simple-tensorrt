FROM nvcr.io/nvidia/tritonserver:22.11-py3

COPY ./models /models

CMD tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs 30
