# Hierarchical Inference Server

This directory holds the programs that shall be run on the server side of the system.

## server_cifar.py & server_imagenet.py

This program implements the server-side of Hierarchical Inference, where images are offloaded to a central server to perform inference when the edge device is not confident in its predictions. The server listens to incoming connections and receives images from edge devices via TCP. It utilizes the ONNX runtime to perform inference using a large model (VIT-H14 for CIFAR-10, ConvNeXT for ImageNet-1k) and returns the result to the edge device. Additionally, the server is configured to use CUDA graphs to optimize performance by avoiding loading CUDA kernels and drivers in each run. The sockets are established and maintained in an open state to enable continuous communication without the necessity of configuring the connection for each communication instance. Any server should have the ability to handle a big number of concurrent TCP connections so that exhaustion of this resource is not a concern. The expense of maintaining these links is modest, as the only additional overhead is the keepalive signal, which by default is set to be transmitted in no less than 2 hours.

### Prerequisites

Before running this program, ensure you have the following installed:

- Python 3.x
- PyTorch
- [ONNX runtime GPU](https://onnxruntime.ai/docs/install/)
- NumPy
- Pillow (PIL)
- torchvision

This program was tested using the **ViT-H/14** model and **ConvNeXT**. You should modify the line `session = ort.InferenceSession("../path/to/vith14.onnx", session_options, providers=providers)` to the specific location of your model.

## server_latency.c

This program implements a simple TCP server that receives the offloaded image from an edge device and immediately sends back a single byte. It is used just for measuring WiFi/Ethernet speeds and latency. Compile and run with:

```bash
gcc -O3 server_latency.c -o server
./server
```
