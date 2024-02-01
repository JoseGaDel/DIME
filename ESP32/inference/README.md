# Inference timing measurements

This project has been built from the [Espressif TensorFlow
Lite for Microcontrollers example]([https://www.tensorflow.org/lite/microcontrollers](https://github.com/espressif/esp-tflite-micro).
It includes the full end-to-end routine for running inference on the CIFAR-10 dataset, which is fed into the microcontroller by the program which can be found in [ESP32/utilities/data_stream.py](https://github.com/JoseGaDel/DIME/tree/main/ESP32/utilities) and has to be run along side this.

### Prerequisites

ESP-IDF needs to be installed to be able tu run this software. For this, follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :

 * The `IDF_PATH` environment variable is set
 * `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`


### Building the project

Assuming the previous condition is met, in this directory the first step is to load ESP-IDF's tools with the path where we have installed esp. For example:

```bash
. $HOME/esp/esp-idf/export.sh
```

This is not necessary if the alias has been already set for that script. To set up the configuration, there is the command

```
idf.py menuconfig
```

The file `sdkconfig.defaults` will load the required configuration for maximal performance, given that by default some configurations may be suboptimal, like the CPU frequency being locked to 160 MHz instead of 240 MHz. Then build with

```
idf.py build
```

### Load and run the program

To flash (replace `/dev/ttyUSB0` with the device serial port):

```
idf.py -p /dev/ttyUSB0 flash
```
Remember not to open the serial monitor to avoid interference with the companion program.

Use `Ctrl+]` to exit. The results from the experiment will be displayed in the terminal running the python program `data_stram.py`
