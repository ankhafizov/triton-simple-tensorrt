import tritonclient.http as tritonhttpclient
import numpy as np
from skimage import io
from skimage.transform import resize

from time import time


VERBOSE = False
input_name = "input_1"
output_name = "predictions"
model_name = "resnet50"
url = "localhost:8000"
model_version = "1"
target_dtype = np.float16


BATCH_SIZE = 32
img = resize(
    io.imread("https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"),
    (224, 224),
)
input_batch = 255 * np.array(
    np.repeat(
        np.expand_dims(np.array(img, dtype=np.float32), axis=0), BATCH_SIZE, axis=0
    ),
    dtype=np.float32,
)

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(
    model_name=model_name, model_version=model_version
)
model_config = triton_client.get_model_config(
    model_name=model_name, model_version=model_version
)

input_batch = input_batch.astype(target_dtype)

times = []
for i in range(100):
    st = time()
    input0 = tritonhttpclient.InferInput(input_name, (32, 224, 224, 3), "FP16")
    input0.set_data_from_numpy(input_batch, binary_data=True)
    output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=True)
    response = triton_client.infer(
        model_name, model_version=model_version, inputs=[input0], outputs=[output]
    )
    logits = response.as_numpy("predictions")
    logits = np.asarray(logits, dtype=np.float32)

    response = triton_client.infer(
        model_name, model_version=model_version, inputs=[input0], outputs=[output]
    )
    times.append((time() - st) * 1000)

print(f"inferense time trt: mean {np.mean(times):.1f} ms, std {np.std(times):.1f} ms")
