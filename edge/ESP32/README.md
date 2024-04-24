# ESP32

### Prerequisites

ESP-IDF needs to be installed to be able to run any of the applications in this project. For this, follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

To make use of this repository, make sure the framework has been installed and the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables):

 * The `IDF_PATH` environment variable is set
 * `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`

It is recommended to follow [Espressif's guide](https://docs.espressif.com/projects/esp-idf/en/v3.2.3/get-started/add-idf_path-to-profile.html), but in Linux and MacOS it would be done by adding 

```bash
export IDF_PATH=~/esp/esp-idf
```

to `~/.profile` or, if you have `/bin/bash` set as login shell, and both `.bash_profile` and `.profile` exist, then update `.bash_profile` instead. For Windows, follow the gide provided above. 

After this, the procedure to build and upload any of the projects is the same.

### Building a project

Assuming the previous condition is met, in the directory of the desired application the first step is to load ESP-IDF's tools with the path where we have installed esp. For example:

```bash
. $HOME/esp/esp-idf/export.sh
```

This is not necessary if the alias has been already set for that script. You may need to perform a clean procedure to ensure there are no residual files from the original project. To do that, run `idf.py fullclean`. To set up the configuration, there is the command

```bash
idf.py menuconfig
```

which will allow us to manually change the configuration of the device. Most projects will have a file named `sdkconfig.defaults` which will load the required configuration for maximal performance, given that by default some configurations may be suboptimal, like the CPU frequency being locked to 160 MHz instead of 240 MHz. When we are satisfied with the set up, build with

```bash
idf.py build
```

and upload with 

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

or you can just use `idf.py -p /dev/ttyUSB0 flash` if you don't want to open the serial monitor, which may be required like in the inference project. If the serial monitor has been opened, use `Ctrl+]` to exit. To find which MCU port is assigned to the ESP32, in Linux you can run the command

```bash
ls /dev/tty*
```

By running this command with and without the ESP32 connected, we can find our device port by observing which entry changes between both runs. You can also run the program: `find_port.sh`
