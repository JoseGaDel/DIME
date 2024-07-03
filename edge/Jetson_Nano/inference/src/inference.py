#!/usr/bin/python3

import tensorrt as trt
import numpy as np
import os
import subprocess
#from timeit import default_timer as timer

import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import ctypes
import pickle

import numpy as np
import os
from PIL import Image
import argparse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self, x:np.ndarray, batch_size=1):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        #self.context.execute(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]



def get_input():
    print(f'Select a model to test:\n   1) Resnet18\n   2) Resnet50\n   3) Alexnet\n')
    choice = int(input("Choice (1/2/3): "))
    print(f'Select precision:\n  1) FP32\n   2) FP16\n')
    precision = int(input("Choice (1/2): "))

    if choice == 1:
        model_path = "../models/resnet18_imagenet"
    
    elif choice == 2:
        model_path = "../models/resnet50_imagenet"
        
    elif choice == 3:
        model_path = "../models/alexnet_imagenet"
    else:
        print("Incorrect input")
        model_path = get_input()

    if precision == 1:
        model_path += "_fp32"
    elif precision == 2:
        model_path += "_fp16"
    else:
        print("Incorrect input")
        model_path = get_input()
    return model_path


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x -= mean
    x /= std
    return x


def LR(array, w1, w2, beta):
    # Find the indices of the two largest elements
    top_indices = np.argpartition(array, -2)[-2:]
    
    # Sort the top two elements
    sorted_indices = np.argsort(array[top_indices])
    
    # Select the two largest elements and apply logistic regression
    x = np.array([array[top_indices[sorted_indices[-1]]], array[top_indices[sorted_indices[-2]]]])
    z = np.dot(np.array([w1, w2]), x) + beta
    probability = 1 / (1 + np.exp(-z))
    
    return probability


if __name__ == "__main__":

    # command = f"python3 setup.py build_ext --inplace"
    # subprocess.run(command, shell=True)
    model_path = get_input()
 
    batch_size = 1
    trt_engine_path = model_path + ".engine"
    
    # Check if the TRT engine file exists
    if not os.path.exists(trt_engine_path):
        # Command to compile the engine if it doesn't exist
        command = f"trtexec --onnx="+model_path+".onnx --fp16 --useCudaGraph --separateProfileRun --precisionConstraints=obey --allowGPUFallback --workspace=3800 --buildOnly --saveEngine="+trt_engine_path

        # Execute the command
        subprocess.run(command, shell=True)
    
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)
    # benchmark(model, input_shape=shape, nruns=100)
    
    correct_predictions = 0
    num_exp = 20000
    label_file = '../data/imagenet_labels.txt'
    image_dir = '../data/imagenet/'
    with open(label_file, 'r') as f:
        labels = f.readlines()

    
    for i in range(num_exp):
        label = labels[0].strip()  # Remove newline character
        image_filename = f"ILSVRC2012_val_{str(i + 1).zfill(8)}.png"
        image_path = os.path.join(image_dir, image_filename)
        data = preprocess_image(image_path)
        scores = model(data, batch_size)
        predicted_label = np.argmax(scores[0][0])
        if int(label) == predicted_label:
            correct_predictions += 1
    
    
    accuracy = correct_predictions / num_exp
    print(f"Accuracy: {accuracy}")