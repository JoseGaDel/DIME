import numpy as np
import os

def preprocess(image, model):
    # Preprocess the image for the specific model
    # Args:
    #     image: image to preprocess
    #     model: model to preprocess the image for
    # Returns:
    #     preprocessed_image: preprocessed image
    if model == "resnet8":
        return image
    elif model == "resnet56":
        std = [0.2023, 0.1994, 0.2010]
        mean = [0.4914, 0.4822, 0.4465]
        preprocessed_image = image / 255.
        preprocessed_image = (preprocessed_image - mean) / std
    elif model == "alexnet":
        std =[62.9932, 62.0887, 66.7048]
        mean = [125.307, 122.95, 113.865]
        preprocessed_image = (image - mean) / std
    return preprocessed_image


def unpickle(file):
    # function to load the CIFAR-10 dataset without preprocessing
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def load_images(path):
    # Load images from a given path
    # Args:
    #     path: path to the images
    # Returns:
    #     images: list of images
    images = []
    for i in range(10000):
        with open(path + "/" + "test_image_"+str(i)+".bin", 'rb') as f:
            # Read binary data
            binary_data = f.read()
            # Convert binary data to numpy array
            image_data = np.frombuffer(binary_data, dtype=np.uint8)
            image_data = np.reshape(image_data, (32, 32, 3))
            
        images.append(image_data)
    return images



def load_images_resnet56(path):
    # Load images from a given path
    # Args:
    #     path: path to the images
    # Returns:
    #     images: list of images
    std = (0.2023, 0.1994, 0.2010)
    mean = (0.4914, 0.4822, 0.4465)

    images = []
    for i in range(10000):
        with open(path + "/" + "test_image_"+str(i)+".bin", 'rb') as f:
            # Read binary data
            binary_data = f.read()
            # Convert binary data to numpy array
            image_data = np.frombuffer(binary_data, dtype=np.uint8)
            image_data = np.reshape(image_data, (32, 32, 3))
            image_data = image_data / 255.
            image_data = (image_data - mean) / std
            
        images.append(image_data)
    return images



def load_images_alexnet(path):
    # Load images from a given path
    # Args:
    #     path: path to the images
    # Returns:
    #     images: list of images
    std =[62.9932, 62.0887, 66.7048]
    mean = [125.307, 122.95, 113.865]

    images = []
    for i in range(10000):
        with open(path + "/" + "test_image_"+str(i)+".bin", 'rb') as f:
            # Read binary data
            binary_data = f.read()
            # Convert binary data to numpy array
            image_data = np.frombuffer(binary_data, dtype=np.uint8)
            image_data = np.reshape(image_data, (32, 32, 3))
            image_data = (image_data - mean) / std
            
        images.append(image_data)
    return images
