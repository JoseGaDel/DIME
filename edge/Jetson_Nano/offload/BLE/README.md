# BLE Central-Peripheral Project

This project implements a Bluetooth Low Energy (BLE) central-peripheral architecture to perform the offload. Here, one device acts as a GATT server, sending a byte array, which would be the image, to a client using notifications. We use notifications to reduce the latency that would be introduced by polling. The client sends back the predicted class from inference using Write Without Response (WWR), which is faster than a Write Request. This option more reliable since it requires the client to send an attribute protocol (ATT) request PDU to the server, followed by a response PDU from the server to the client, but introduces a higher latency as both ends need to acknowledge the data transfer. In our experiments, it was observed that WWR was significantly faster than a Write Request and no issue with data integrity was noted, which is why we chose to use it. The server then receives the predicted class and prints it to the console.

## Implementation Details

This project uses Python API for BlueZ, the Linux Bluetooth stack, in both ends. Therefore, both machines need to run on Linux. This implementation is based on the examples in  [Bluetooth for Linux Developers Study Guide](https://www.bluetooth.com/blog/the-bluetooth-for-linux-developers-study-guide/), which itself is based on the [official BlueZ documentation samples](https://git.kernel.org/pub/scm/bluetooth/bluez.git/tree/doc). The directory on the raspberry is an exact copy of this. To maximize performance, you may consider changing some parameters in the bluetooth configuration. Some of them are:

- Connection interval: The time between two consecutive connection events. The default value is 30ms, but it can be changed to 7.5ms, 15ms, 30ms, 60ms, 100ms, 200ms, 400ms, 800ms, 1600ms, or 3200ms. The lower the value, the faster the connection, but it may consume more power. You can run the following commands to change the connection interval:

```bash
sudo sh -c "echo  8 > /sys/kernel/debug/bluetooth/hci0/conn_min_interval"
sudo sh -c "echo  12 > /sys/kernel/debug/bluetooth/hci0/conn_max_interval"
```

- Physical Layer: The physical layer can be changed to 1M or 2M. The default value is 1M. You can run the following command to change the physical layer:

```bash
sudo hcitool cmd 0x08 0x0031 0x00 0x02 0x02
```

Other configurations may be changed, but only those two have been tested in our program. Another important detail is to make sure the device has bluetooth capabilities version 5.0 or higher, as lower versions will experience a significant increase in latency. To check your bluetooth version, you can run the following command:

```bash
hciconfig -a
```

This will also give you the MAC address of the device, which is necessary to run the program (you can also run `hcitool dev`).

## Running the Program

To run the program, you need to run the server first. The server will start advertising the service and wait for a connection. To run the server, you need to run the following command:

```bash
python server.py
```

After the server is running, you can run the client. The client will scan for the server and connect to it. To run the client, you need to run the following command:

```bash
python client.py <server_mac_address>
```

You may get prompted that the devices could not find each other. If so, you can run:

```bash
bluetoothctl
[bluetooth]# scan le
```

and once you see a line with something like:

```
[NEW] Device MAC:ADDRESS Name: HI Central
```

Then you are good to go. The program will send an image for a certain number of iterations, which can be changed in the program, and save the latencies so you can get the mean and standard deviation.
