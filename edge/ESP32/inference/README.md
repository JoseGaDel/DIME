# Inference timing measurements

This project has been built from the [Espressif TensorFlow
Lite for Microcontrollers example](https://github.com/espressif/esp-tflite-micro).
It includes the full end-to-end routine for running inference on the CIFAR-10 dataset, which is fed into the microcontroller by the program which can be found in [ESP32/utilities/data_stream.py](../utilities/) and has to be run along side this.

## Building the project

Assuming the enviroment has been set up as explained in the README.md of the parent directory, the building procedure is as explained in that same document. When flashing the program, remember not to open the serial monitor to avoid interference with the companion program. The results from the experiment will be displayed in the terminal running the python program `data_stram.py`. 

## Program structure

In the main directory, the main program is in `main_fucntions.cc`. The only manual adjustment that may be needed is in the value of the macro `#define BAUD_RATE` which must match the value in `data_stream.py`. The program takes the weights stored in `model.cc`, builds the CNN with TensorFlow Lite and performs inference using the stream of images sent by the python program. A logistic regression model has been trained to predict when the model fails to recognize successfully the class of an image. This implementation can be found in `LR.h`, and the parameters `weight1`, `weight2`, and `beta` are hardcoded into the program itself.
