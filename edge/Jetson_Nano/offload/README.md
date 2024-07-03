# Image Transfer Benchmark

This program measures the latency and transfer speeds of sending images to a server and receiving a response. It is designed to work with the CIFAR-10 and ImageNet-1k datasets, sending images as 8-bit unsigned integer arrays over a TCP connection.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/JoseGaDel/DIME.git
cd edge/Jetson_Nano/offload/
```

2. Set up the CMake project and compile. You need to indicate if you want to measure latency or energy consumption by setting the `LATENCY` and `ENERGY` flags to 1 or 0, respectively. For example, to measure latency and not energy consumption, run the following command:

```bash
cmake -DLATENCY=1 -DENERGY=0 .
```

3. Run the program [DIME/server/server_latency.c](https://github.com/JoseGaDel/DIME/blob/main/server/) on the server side.

4. Run the client program on the edge device:

```bash
./client <dataset>
```

We need to specify the dataset to use. The program supports either CIFAR-10 or Imagenet1k. To use CIFAR-10, we have to pass `cifar` (or `cifar10`, `CIFAR-10`, `cifar-10`) as an argument. For Imagenet1k, we have to pass `imagenet1k` (or a couple more similar ways you can check in the code).

## Notes
* The program sends images expects data can be found in the `data/` directory found in the `inference` section of this device. You may change the direction of such directory in the macro `DATA_PATH`, as well as the IP address and port of the server in the macros `SERVER_IP` and `SERVER_PORT`, respectively.
* Latency and transfer speeds are measured for each image sent.
* Transfer speed is calculated based on the size of the image sent and the transfer time. If the server performs any other operations between receiving and sending, this number may not reflect the actual transfer speeds accurately.
* The program uses raw 8-bit unsigned integers to represent images. Adjustments may be needed if your model requires a different format.
* Ensure that the server is running and listening on the specified IP address and port before running the program.
