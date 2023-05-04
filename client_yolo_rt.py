# YOLO RT: https://github.com/zhiqwang/yolov5-rt-stack

from time import time

import tritonclient.http as tritonhttpclient
import cv2
import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import scale_boxes, check_img_size


def preprocess_img(img_input, img_size):
    img = img_input.copy()

    img_size = check_img_size(img_size, s=32)
    img = letterbox(img, (img_size, img_size), auto=False)[0]

    img = img.transpose((2, 0, 1))[::-1]
    new_img = np.divide(img, 255)
    new_img = new_img[None]
    return new_img


# ---------------------------------- Params ----------------------------------

img_size = 640
img_orig = cv2.imread("data/zidane.jpg")

img = preprocess_img(img_orig, img_size)

input_name = "images"
output_names = [
    "detection_boxes",
    "detection_classes",
    "detection_scores",
    "num_detections",
]

model_name = "yolo5mrt"
url = "localhost:8000"
model_version = "1"
VERBOSE = False

# -------------------------------- Inferense ---------------------------------

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)

times = []
for i in range(100):
    st = time()
    input0 = tritonhttpclient.InferInput(input_name, img.shape, "FP32")
    input0.set_data_from_numpy(img.astype(np.float32))
    outputs = [tritonhttpclient.InferRequestedOutput(n) for n in output_names]
    response = triton_client.infer(
        model_name, model_version=model_version, inputs=[input0], outputs=outputs
    )
    times.append((time() - st) * 1000)
print(f"inferense time trt: mean {np.mean(times):.1f} ms, std {np.std(times):.1f} ms")

# ---------------------------- Parcing results -------------------------------

# results = {}

# num_detections = response.as_numpy("num_detections")[0][0]
# for name in ["detection_boxes", "detection_classes", "detection_scores"]:
#     results[name] = response.as_numpy(name)[0][:num_detections]

# for bbx, lbl, conf in zip(
#     results["detection_boxes"],
#     results["detection_classes"],
#     results["detection_scores"],
# ):
#     bbx = scale_boxes((img_size, img_size), bbx, img_orig.shape)

#     x1, y1, x2, y2 = bbx.astype(int)
#     img_orig = cv2.rectangle(img_orig, (x1, y1), (x2, y2), [0, 0, 255], 2)
#     img_orig = cv2.putText(
#         img_orig, str(lbl), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, [0, 0, 255], 2
#     )

# cv2.imshow("zidane", img_orig)
# cv2.waitKey(0)
