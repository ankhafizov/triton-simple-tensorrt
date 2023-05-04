FROM nvcr.io/nvidia/tritonserver:22.11-py3

COPY ./model_repository /model_repository

CMD tritonserver --model-repository=/model_repository --model-control-mode=poll --repository-poll-secs 30
