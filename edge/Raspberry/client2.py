'''
    This program implements the edge side of Hirearchical Inference, where images are offloaded to a
    central server to perform inference on the images where the edge device is not confident to have
    performed a good prediction. The client establishes a TCP connection with the central server,
    performs inference using the TinyML model, runs logistic regression to determine if the result
    should be accepted or offloaded, sends the image to the server and waits for it's response.
    The program measures the latency of the edge side, the server side and the accuracies of both.
'''

import socket
import tensorflow as tf
import numpy as np
import time
import argparse
from utils import preprocess, unpickle



# Load the data batch
data_batch = unpickle('data/test_batch')

# Access the data and labels
images = data_batch[b'data']  # Images
labels = data_batch[b'labels']  # Labels



def main(args):
    # Access the arguments
    model_path = args.path
    model = args.model
    mode = args.mode
    try:
        num_images = int(args.num_images)
    except AttributeError:
        num_images = 10000
        print("Defaulting to the maximum number of images: 10000")

    # Print the selected arguments
    print("Model file:", model_path)
    print("Model configuration:", model)
    print("Program mode:", mode)
    print("Number of images:", num_images)

    if model == "resnet8":
        beta, w1, w2 = +4.53162290, -3.59687685, 5.82555453
    
    elif model == "resnet56":
        beta, w1, w2 = +4.50152442, -3.943078, 5.02825697 
    
    elif model == "alexnet":
        beta, w1, w2 = +3.79694817, -1.79537036, 4.84755251
    else:
        raise ValueError("Incorrect model. Please select a valid model: resnet8, resnet56, alexnet")
    
    if mode == "test":
        always_offload = True
    elif mode == "benchmark":
        always_offload = False
    else:
        raise ValueError("Incorrect mode. Please select a valid mode: test, benchmark")
    
    if num_images > 10000:
        num_images = 10000
        print("The number of images is too high. Defaulting to the maximum number of images: 10000")
    elif num_images < 1:
        num_images = 1
        print("The number of images is too low. Defaulting to the minimum number of images: 1")


    
    # Server configuration
    SERVER_HOST = 'uranus.networks.imdea.org'  # Replace with the IP address or domain of your server
    SERVER_PORT = 12345

    responses = []
    true_labels = []
    inference_timmings = []
    offload_timmings = []
    total_timmings = []

    correct_edge = 0
    correct_server = 0
    correct_edge_off = 0

    #Load the model
    with open(model_path, 'rb') as fid:
        tflite_model = fid.read()

    # setup TF Lite
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # small warmup
    for i in range(5):
        interpreter.set_tensor(input_details[0]['index'], np.float32(images[i]).reshape(1,32,32,3)) ### the type of the input (and -128) changes if the model is not quantized
        interpreter.invoke() 
        output = interpreter.get_tensor(output_details[0]['index'])

    # Start the TCP communication with the server
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    # Send image data to the server
    for i in range(num_images):
        # Reshape the image to 32x32x3 (RGB channels) and permute the indices to 3x32x32
        image =  preprocess(images[i], model)
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        # Add extra dimension to get the expected input shape 1x3x32x32
        expanded_image = np.expand_dims(image, axis=0)
        # start measuring time and perform inference
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], np.float32(expanded_image)) ### the type of the input (and -128) changes if the model is not quantized
        interpreter.invoke() 
        output = interpreter.get_tensor(output_details[0]['index'])
        clase = np.argmax(output)
        # Perform logistic regression to decide the validity of the inference
        offload = round(1/ (1 + np.exp(beta - np.dot([w1, w2],np.partition(output[0], -2)[[-2,-1]]))))
        inference_timmings.append(time.time() - start)
        if offload <= 0.5 or always_offload:
            # send image to the server
            start_offload = time.time()
            client_socket.sendall(images[i])
            # Receive response from the server
            response = client_socket.recv(8)
            offload_timmings.append(time.time() - start_offload)
            if int.from_bytes(response,"little") == labels[i]:
                correct_server += 1

        total_timmings.append(time.time() - start)

        if clase == labels[i]:
            if offload > 0.5 and not always_offload:
                correct_edge_off += 1
            correct_edge += 1

            


    avg_total = sum(total_timmings)/len(total_timmings)
    avg_inference = sum(inference_timmings)/len(inference_timmings)
    avg_offload = sum(offload_timmings)/len(offload_timmings)
    server_precission = correct_server*100/len(offload_timmings)
    edge_precission = correct_edge*100/len(inference_timmings)
    system_precission = (correct_edge_off + correct_server)*100/num_images

    if mode == "test":
        print(f'\nInference time:     {avg_inference*1000} ms\nOffload time:       {avg_offload*1000} ms\nTotal time:         {avg_total*1000} ms\nServer precission:  {server_precission} %\nEdge precission:    {edge_precission} %\n')
    else:
        print(f'\nInference time:     {avg_inference*1000} ms\nOffload time:       {avg_offload*1000} ms\nTotal time:         {avg_total*1000} ms\nServer precission:  {server_precission} %\nEdge precission:    {edge_precission} %\nSystem precission:  {system_precission} %\n')

    # Close the connection
    client_socket.close()


# Initialize ArgumentParser
parser = argparse.ArgumentParser(description="Description of the program.",
                                 usage=f"\
python3 client2.py <model_path> <model> <mode> <number_of_images>\n\n\
    • model_path: Path to the model file\n\
    • model:\n\t  - resnet8\n\t  - resnet56\n\t  - alexnet\n\
    • mode:\n\t  - test  -> All runs are offloaded\n\t  - benchmark  -> Realistic scenario. Offload when LR predicts wrong inference\n\
    • number_of_images: Number of images to process (default 10000)\n\n")

# Add arguments
parser.add_argument("path", help="path/to/the/model/file")
parser.add_argument("model", help="Specify the model (resnet8, resnet56, alexnet)")
parser.add_argument("mode", help="Specify the mode (test, benchmark)")
parser.add_argument("num_images", help="Specify the number of images to perform the test (1 to 10000)")

# Parse the arguments
args = parser.parse_args()

# Call the main function with parsed arguments
main(args)
