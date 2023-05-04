import torch
from time import time
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='onnx/yolov5m6.pt', force_reload=True)

times = []
for i in range(100):
    st = time()
    results = model("data/zidane.jpg")
    # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    times.append((time() - st) * 1000)

print(f"inferense time trt: mean {np.mean(times):.1f} ms, std: {np.std(times):.1f} ms")
