# Image Transfer Benchmark

This program measures the latency and transfer speeds of sending images to a server and receiving a response. It is designed to work with the CIFAR-10 dataset (although some variables like the image width, height and channels can be changed for different datasets as long as they are stored in a similar way) sending images as 8-bit unsigned integer arrays over a TCP connection.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/JoseGaDel/DIME.git
cd edge/Jetson_Nano/wifi/
```

2. Create a build directory, set up the CMake project and compile:

```bash
mkdir build
cd build
cmake ..
make
```

3. Run the server program in 

4. Run the client program on the edge device:

```bash
./TCPClient <number_of_runs>
```

The program allows us to select how many images we want to send to the server to measure latency and transfer times by passing such number as an argument. If we do not pass anything, the program will send the entire dataset by calculating how many images it contains based on the size of an individual image declared at the beginning of the program with `const int width = 32, height = 32, channels = 3;` and the size of the file.

## Notes
* The program sends images from the CIFAR-10 dataset to a server specified by the IP address and port number in the code.
* Latency and transfer speeds are measured for each image sent.
* Transfer speed is calculated based on the size of the image sent and the transfer time. If the server performs any other operations between receiving and sending, this number may not reflect the actual transfer speeds accurately.
* The program uses raw 8-bit unsigned integers to represent images. Adjustments may be needed if your model requires a different format.
* Ensure that the server is running and listening on the specified IP address and port before running the program.
