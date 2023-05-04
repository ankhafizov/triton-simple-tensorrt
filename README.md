# Triton Tensorrt

Простой пример для использования tensorrt совместно с triton infernce server

источники:

- https://towardsdatascience.com/serving-tensorrt-models-with-nvidia-triton-inference-server-5b68cc141d19

- https://github.com/tanpengshi/NVIDIA_Triton_Server_TensorRT


# Конвертация YOLO:

1. docker pull nvcr.io/nvidia/tensorrt:22.11-py3

2. docker run -d -it --gpus all -v ${PWD}/onnx:/trt_optimize nvcr.io/nvidia/tensorrt:22.11-py3

3. /workspace/tensorrt/bin/trtexec --onnx=yolov5m.onnx --saveEngine=yolo5m.plan  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16

либо использовать: https://github.com/ankhafizov/yolov5-rt-stack