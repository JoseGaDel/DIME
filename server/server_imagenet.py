'''
    This program implements the server side of Hierarchical Inference, where images are offloaded to a
    central server to perform inference on the images where the edge device is not confident to have
    performed a good prediction. The server listens to incoming connections and receives images from the
    edge device via TCP. It then uses ONNX runtime to perform inference using a big model (VIT-H14) and
    returns the result to the edge device. The server is also configured to use CUDA graphs to optimize
    to avoid loading CUDA kernels and drivers in each run.
'''

import socket
import torchvision.transforms as transforms
import torch
import onnxruntime as ort
import onnx
import numpy as np
from PIL import Image

# Server configuration
SERVER_HOST = '0.0.0.0'  # Listen on all network interfaces
SERVER_PORT = 12345

image_size = 224 * 224 * 3

# Define the image transformation using torchvision. The image needs to be upscaled to 224x224 pixels 
# and normalized with pixel = (pixel - 0.5) / 0.5.
transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the numpy.ndarray has dtype = np.uint8
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
preprocess = torch.compile(transform, mode="max-autotune")

model_name = "imagenet_facebook.onnx"
onnx_model = onnx.load(model_name)

providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648*15,
        'trt_fp16_enable': True,
        'trt_builder_optimization_level': 5,
        'trt_engine_cache_enable' : True,
        'trt_dump_ep_context_model' : True,
        'trt_ep_context_file_path' : './',
        'trt_cuda_graph_enable': True,
        'trt_layer_norm_fp32_fallback': True,
    }),
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": 'kNextPowerOfTwo',
        # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": 'EXHAUSTIVE',
        "do_copy_in_default_stream": True,
        "enable_cuda_graph": 1,
        "cudnn_conv_use_max_workspace": '1',
        "cudnn_conv1d_pad_to_nc1d": '1',
        #"prefer_nhwc": '1'
    })
]


# Create a session options object and set the providers with the configuration above.
session_options = ort.SessionOptions()

# session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# session_options.optimized_model_filepath = model_name.replace('.onnx', '') + '_opt.onnx'
# session = ort.InferenceSession(onnx_model.SerializeToString(), session_options)

# Load the ONNX model
session = ort.InferenceSession(onnx_model.SerializeToString(), session_options, providers=providers)
#session = ort.InferenceSession(onnx_model, session_options, providers=providers)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name

print(f'input shape: {session.get_inputs()[0].shape}\noutput shape: {session.get_outputs()[0].shape}\n')

# To implement CUDA graphs for the model, we need to create the input and output tensors and bind them to the session.
# we create a test input and output tensor and run one inference for CUDA to capture the graph that will be used
# in each inference.
inp = np.random.rand(*session.get_inputs()[0].shape)        # Values between 0.0 and 1.0
inp = inp.astype(np.float32)
out = np.random.rand(*session.get_outputs()[0].shape)       # Values between 0.0 and 1.0
out = out.astype(np.float32)
out = np.zeros(session.get_outputs()[0].shape, dtype=np.float32)
input_tensor = ort.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
output_tensor = ort.OrtValue.ortvalue_from_numpy(out, 'cuda', 0)
io_binding = session.io_binding()

# Pass gpu_graph_id to RunOptions through RunConfigs
ro = ort.RunOptions()
ro.add_run_config_entry("gpu_graph_id", "0")

# Bind the input and output
io_binding.bind_ortvalue_input(input_name, input_tensor)
io_binding.bind_ortvalue_output(label_name, output_tensor)

# One regular run for the necessary memory allocation and cuda graph capturing
session.run_with_iobinding(io_binding)

# we make a small warmup to avoid the first inference to be slow
warmup = 100
print(f'____ Performing warmup ____\n')
for i in range(warmup):
    session.run_with_iobinding(io_binding, ro)
    print(f'warmup {i}/{warmup}...', end='\r')


reception = []

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((SERVER_HOST, SERVER_PORT))

# Listen for incoming connections
server_socket.listen(1)  # Allow up to 5 connections to wait in the queue
print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

while True:
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    #print(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")
    socket_closed =  False
    # Keep receiving images until the client closes the connection
    while True:
        image_data = []
        while True:
            chunk = client_socket.recv(image_size)
            if not chunk:
                socket_closed =  True
                break
            image_data += chunk
            if len(image_data) == image_size:
                break
        if socket_closed:
            break
        # Unflatten the image data and transform it to a tensor
        image_array = np.array(image_data, dtype=np.uint8)
        image_array = image_array.reshape(3, 224, 224)#.transpose(1, 2, 0)
        # torchvision expects a PIL image
        image = Image.fromarray(image_array)
        # Apply the preprocessing
        image = transform(image)
        
        # Send image to the input buffer
        input_tensor.update_inplace(image.numpy())
        # Perform inference using the captured CUDA graph
        session.run_with_iobinding(io_binding, ro)
        # Copy the result from device to the host
        output = io_binding.copy_outputs_to_cpu()[0]
        response_message = np.uint8(np.argmax(output))
        # Send response back to the client
        client_socket.sendall(response_message.tobytes())


    # Close the connection
    #client_socket.close()
    #print("socket closed")
