# Hierarchical Inference Server

This directory holds the programs that shall be run on the server side of the system.

## server.py

This program implements the server-side of Hierarchical Inference, where images are offloaded to a central server to perform inference when the edge device is not confident in its predictions. The server listens to incoming connections and receives images from edge devices via TCP. It utilizes the ONNX runtime to perform inference using a large model (VIT-H14) and returns the result to the edge device. Additionally, the server is configured to use CUDA graphs to optimize performance by avoiding loading CUDA kernels and drivers in each run.

### Prerequisites

Before running this program, ensure you have the following installed:

- Python 3.x
- PyTorch
- [ONNX runtime with CUDA EP]([https://onnxruntime.ai/docs/install/)
- NumPy
- Pillow (PIL)
- torchvision

This program was tested using the **ViT-H/14** model. You should modify the line `session = ort.InferenceSession("../path/to/vith14.onnx", session_options, providers=providers)` to the especific localization of your model.
