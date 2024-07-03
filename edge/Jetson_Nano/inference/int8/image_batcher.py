# This program builds from the original image_batcher.py file from the TensorRT samples:
# https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/image_batcher.py
#


import os
import sys

import numpy as np
from PIL import Image


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(
        self,
        input,
        shape,
        dtype,
        max_num_images=None,
        exact_batches=False,
        preprocessor="resnet8",
    ):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used. If the dataset is
        imagenet, the same preprocessing as the original EfficientNet is used, regardless of the value of this parameter.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return (
                os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
            )

        if os.path.isdir(input):
            self.images = [
                os.path.join(input, f)
                for f in os.listdir(input)
                if is_image(os.path.join(input, f))
            ]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes cropping,
        resizing, normalization, data type casting, and transposing.
        :param image_path: The path to the image on disk to load.
        :return: A numpy array holding the image sample, ready to be contacatenated into the rest of the batch.
        """

        def pad_crop(image):
            """
            A subroutine to implement padded cropping. This will create a center crop of the image, padded by 32 pixels.
            :param image: The PIL image object
            :return: The PIL image object already padded and cropped.
            """
            # Assume square images
            assert self.height == self.width
            width, height = image.size
            ratio = self.height / (self.height + 32)
            crop_size = int(ratio * min(height, width))
            y = (height - crop_size) // 2
            x = (width - crop_size) // 2
            return image.crop((x, y, x + crop_size, y + crop_size))

        image = Image.open(image_path)
        image = image.convert(mode="RGB")

        image = np.asarray(image, dtype=self.dtype)
        image = image / 255.0
        image = image - np.asarray([0.485, 0.456, 0.406])
        image = image / np.asarray([0.229, 0.224, 0.225])

        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding two items per iteration: a numpy array holding a batch of images, and the list of
        paths to the images loaded within this batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images




class CIFAR10BinaryBatcher:
    """
    Creates batches of pre-processed CIFAR-10 images from binary files.
    """

    def __init__(self, file_path, shape, dtype, batch_size=32, preprocessor="resnet8"):
        """
        :param file_path: The path to the CIFAR-10 binary file.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param batch_size: The size of the batches to create.
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        """
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self.batch_size = batch_size
        self.preprocessor = preprocessor

        # Load CIFAR-10 binary file
        self.images, self.labels = self.load_cifar10_binary(file_path)

        self.num_images = len(self.images)

        # Handle Tensor Shape
        assert len(self.shape) == 4
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

    def load_cifar10_binary(self, file_path):
        """
        Load CIFAR-10 images and labels from a binary file.
        :param file_path: The path to the CIFAR-10 binary file.
        :return: A tuple containing images and labels as numpy arrays.
        """
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        num_records = len(data) // 3073
        data = data.reshape(num_records, 3073)
        labels = data[:, 0]
        images = data[:, 1:].reshape(num_records, 3, 32, 32)
        
        return images, labels

    def preprocess_image(self, image):
        """
        The image preprocessor prepares the image as needed for batching. This includes normalization,
        data type casting, and transposing.
        :param image: The numpy array holding the image data.
        :return: A numpy array holding the image sample, ready to be concatenated into the rest of the batch.
        """
        image = image.transpose(1, 2, 0)  # Convert from CHW to HWC

        if self.preprocessor == "resnet8":
            image = image.astype(self.dtype)
        elif self.preprocessor == "resnet56":
            image = image.astype(self.dtype)
            image = image / 255.0
            image = image - np.asarray([0.4914, 0.4822, 0.4465])
            image = image / np.asarray([0.2023, 0.1994, 0.2010])
        elif self.preprocessor == "alexnet":
            image = image.astype(self.dtype)
            image = image - np.asarray([125.307, 122.95, 113.865])
            image = image / np.asarray([62.9932, 62.0887, 66.7048])
        elif self.preprocessor == "imagenet":
            image = image.astype(self.dtype)
            image = image / 255.0
            image = image - np.asarray([0.485, 0.456, 0.406])
            image = image / np.asarray([0.229, 0.224, 0.225])
        else:
            print(f"Preprocessing method {self.preprocessor} not supported")
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding two items per iteration: a numpy array holding a batch of images, and the list of
        indices of the images loaded within this batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for j, image in enumerate(batch_images):
                batch_data[j] = self.preprocess_image(image)
            yield batch_data, [self.image_index + j for j in range(len(batch_images))]
            self.batch_index += 1
            self.image_index += self.batch_size




