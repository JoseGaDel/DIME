# Arduino Nano 33 BLE Sense Measurements

## Main program
This folder contains the program used to get the results of the TinyML and LR models. The output is a list containing three values for each sample: index of the image in the given batch, LR output and predicted class. 

To run this program, one needs to feed the desired images via USB connection. The images are stored and processed in batches of 33 (this value can be changes respecting the memory constraints)


## Time measurements
It contains basicallly the same program as "main_program" but provides a different output. In this case given a certain image it provides three values: Total time, inference time and LR time. 

## BLE communication
This folder contains a program used to measure the time needed to send an image to PC and get one byte back representing the predicted class. A python file ("blue.py") is given to run on a central device. This program receives the image and sends back one byte. The board stops the time as soon as the byte is received.
