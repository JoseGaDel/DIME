# Utilities for ESP32 Measurements

This directory contains a set of companion programs designed to be run on your local machine in tandem with programs running on the ESP32.

## Table of Contents

- [Inference](#inference)
- [WiFi communications](#wifi)


## Inference

To measure inference times in the ESP32 using [TinyML' ResNet](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification/trained_models), the program located in [ESP32/inference/](DIME/ESP32/inference/) has to be built and flashed into the MCU. The program `data_stream.py` in this directory needs to be run in a system connected through the serial port to the microcontroller. The program running in the ESP32 will print the data to the serial port so as to avoid wrong conversions when the python program attempts to read raw byte data. Therefore, it is important that the **serial monitor is not open** as it will interfere with communication. Tu run the program, simply run:

```bash
python data_stream.py
```

The program will look if there is a file `data_file.npy` containing the results from a previous run so it can load them and resume from where it was left. If it exist, it will ask if it should load it or start from scratch. The user will be asked how many images are needed to perform inference, and will take the entire dataset if the user just presses enter.

The program may need some setup. At the top of the program, the following lines may be subject to modification for the specific needs of your project:

```pyhton
mcu_port = "/dev/ttyUSB0"
data_path = "./data/cifar-10-batches-bin/test_batch.bin"
baud_rate = 460800
image_size = 3073
```

The MCU port may need special attention. In Linux, you can check the port by running the command

```bash
ls /dev/tty*
```

first with the device disconnected, and then a second time with the device connected. The second time a new entry will appear that wasn't in the first, which is your device. An example of running 3300 images is as follows

| **Good predictions** | **Bad predictions** | **Precision** | **Num. Offloads** | **Correct LR predictions** | **Precision without offload** |
|:--------------------:|:-------------------:|:-------------:|:-----------------:|:--------------------------:|:-----------------------------:|
|         3187         |         113         |    96.5757    |        721        |             322            |            86.8181            |

## Inference
