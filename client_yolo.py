import tritonclient.http as tritonhttpclient
import cv2
import numpy as np
from yolov5.utils.general import non_max_suppression
import torch
from time import time


img = cv2.imread("data/zidane.jpg")
img = cv2.resize(img, (640, 640))
img_orig = img.copy()

img = img.transpose((2, 0, 1))[::-1]
img = img.astype(np.float16)
img /= 255
img = img[None]

input_name = "images"
output_name = "output0"
model_name = "yolo5m"
url = "localhost:8000"
model_version = "1"
VERBOSE = False

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)

times = []
for i in range(100):
    st = time()
    input0 = tritonhttpclient.InferInput(input_name, img.shape, "FP16")
    input0.set_data_from_numpy(img)

    output = tritonhttpclient.InferRequestedOutput(output_name)

    response = triton_client.infer(
        model_name, model_version=model_version, inputs=[input0], outputs=[output]
    )
    results = torch.from_numpy(response.as_numpy("output0"))

    results = non_max_suppression(results, 0.3, 0.4)[0]
    times.append((time() - st) * 1000)

print(f"inferense time trt: mean {np.mean(times):.1f} ms, std: {np.std(times):.1f} ms")

# labels = [lbl for lbl in results[:, -1].numpy()]
# bboxes = list(results[:, :-2].numpy().astype(int))
# confidences = list(results[:, -2].numpy().astype(float))

# for lbl, bbx, conf in zip(labels, bboxes, confidences):
#     x1, y1, x2, y2 = bbx
#     img_orig = cv2.rectangle(img_orig, (x1, y1), (x2, y2), [0, 0, 255], 2)
#     img_orig = cv2.putText(
#         img_orig, str(lbl), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, [0, 0, 255], 2
#     )

# cv2.imshow("zidane", img_orig)
# cv2.waitKey(0)
