# WiFi communication via TCP

This project has been built on top of the ESP-IDF's example to showcase [TCP client configuration](https://github.com/espressif/esp-idf/tree/master/examples/protocols/sockets/tcp_client). The application creates a TCP socket and tries to connect to the server with predefined IP address and port number. When a connection is successfully established, the application sends the image and waits for the answer, which would be a single byte representing the result from inference being run in the server with the offloaded image sent by the MCU. The sockets will remain open for persistent communication, and the application will perform this communication multiple times, established in the macro `MEASUREMENTS`, to average out the communication time. This stems from the inherent variability in Wi-Fi environments, influenced by factors such as congestion, signal noise, and other external interferences, which introduce fluctuations in individual measurements, making them less representative of the true performance of the communication channel. By averaging the roundtrip times across multiple runs, the effect of outliers get dampened.

## Configure the project

The project requires some manual set up to accommodate it to your particular environment. The file `setup.h` in the directory `main` contains some macros to configure the program. In `SSID` and `PASS`, introduce the name of the access point the ESP32 will connect to and its password. If the network is open, leave the string associated to `PASS` empty and the program will automatically set up the authentication mode as open. If the network has some kind of authentication mode rather than the default WPA2, uncomment the desired configuration and leave the rest commented. The next important macro is `HOST_IP_ADDR`, which will store the IP address of the system hosting the companion TCP server program located in [DIME/ESP32/utilities/tcp_server.c](https://github.com/JoseGaDel/DIME/tree/main/ESP32/utilities). To find which IP value to assign, in the machine running the server we can run the command:

```bash
ip addr show
```

The value of `PORT` has to match the port the TCP server will be listening to. The values in the file `sdkconfig.defaults` contain some important values to ensure the TCP communication is performed efficiently. Most of this configuration is taken from the [esp32 iperf example](https://docs.espressif.com/projects/esp-idf/en/v5.0/esp32/api-guides/lwip.html#lwip-performance) that contains settings to maximize TCP/IP throughput. For further information on speed optimization in this context, refer to Espressif's guide on [WiFi performance](https://docs.espressif.com/projects/esp-idf/en/v5.0/esp32/api-guides/wifi.html#how-to-improve-wi-fi-performance) and [WiFi buffer usage](https://docs.espressif.com/projects/esp-idf/en/v5.0/esp32/api-guides/wifi.html#wifi-buffer-usage). We now present a brief justification on some of the settings.

- **TCP Buffer Sizes** (CONFIG_LWIP_TCP_SND_BUF_DEFAULT and CONFIG_LWIP_TCP_WND_DEFAULT): These parameters set the default sizes for TCP send and receive buffers. A larger buffer size reduces the potential for flow control to occur, and results in improved CPU utilization

- **lwIP Task and Mailbox Sizes** (CONFIG_LWIP_TCP_RECVMBOX_SIZE, CONFIG_LWIP_UDP_RECVMBOX_SIZE, CONFIG_LWIP_TCPIP_RECVMBOX_SIZE): These configurations adjust the sizes of mailboxes used by lwIP tasks for TCP communications. Proper sizing ensures efficient handling of incoming packets, reducing the likelihood of overflow and improving overall performance.

- **lwIP IRAM Optimization** (CONFIG_LWIP_IRAM_OPTIMIZATION and CONFIG_LWIP_EXTRA_IRAM_OPTIMIZATION): Enabling IRAM (Internal RAM) optimization directs lwIP code to be stored in the faster and more limited internal RAM of the ESP32. This can improve the execution speed of lwIP functions, enhancing overall TCP communication performance.

- **lwIP TCP/IP Task Priority** (CONFIG_LWIP_TCPIP_TASK_PRIO): This sets the priority of the lwIP TCP/IP task. A higher priority ensures that lwIP tasks are processed promptly, potentially reducing latency in TCP communication.

- **lwIP Core Locking** (CONFIG_LWIP_TCPIP_CORE_LOCKING and CONFIG_LWIP_TCPIP_CORE_LOCKING_INPUT): Enabling core locking ensures that lwIP functions are thread-safe. It's crucial for environments where the TCP/IP stack might be accessed by multiple tasks concurrently.

- **Wi-Fi Buffer Configuration** (CONFIG_ESP32_WIFI_STATIC_RX_BUFFER_NUM, CONFIG_ESP32_WIFI_DYNAMIC_RX_BUFFER_NUM, CONFIG_ESP32_WIFI_DYNAMIC_TX_BUFFER_NUM, CONFIG_ESP32_WIFI_TX_BA_WIN, CONFIG_ESP32_WIFI_RX_BA_WIN, CONFIG_ESP32_WIFI_AMPDU_TX_ENABLED, CONFIG_ESP32_WIFI_AMPDU_RX_ENABLED): These parameters configure the Wi-Fi buffer sizes, block acknowledgment (BA) window sizes, and enable/disable mechanisms like A-MPDU (Aggregated MPDU). Proper tuning of these parameters can enhance the efficiency of Wi-Fi communication.

If the command `idf.py menuconfig` is run, the maximum and minimum values those parameters can be inspected.

## Program structure

Apart from the aforementioned optimizations, the program `tcp_client_main.c` contains some additional optimizations in the configuration of the WiFi communication. For instance, in the function `connect_wifi()`,  the following lines ensure some WiFi driver directives are active to increase performance:

```C
// set transmission mode to 40 MHz
ESP_ERROR_CHECK(esp_wifi_set_bandwidth(0, WIFI_BW_HT40));

// disable power saving mode
ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

// set WiFi protocol
ESP_ERROR_CHECK(esp_wifi_set_protocol(0, WIFI_PROTOCOLS));

// store WiFi configuration in RAM. Can impact performance
ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

// set the wifi config
ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config) );
```

The first call sets the Wi-Fi transmission mode to use a 40 MHz channel width. Wider channel widths can increase data throughput, as more frequency spectrum is utilized. The second call disables power saving mode (WIFI_PS_NONE). Power saving features may introduce latency when the Wi-Fi module is in a low-power state and needs to wake up. The third uses the macro `WIFI_PROTOCOLS` defined in `setup.h` which contains a bitmask specifying which protocols to enable. It might includes support for 802.11b, 802.11g, and/or 802.11n. Adjusting the protocols can impact compatibility and performance. If we add the following block after `esp_wifi_start()`: 

```C
uint8_t getprotocol;
esp_err_t errwf;
errwf = esp_wifi_get_protocol(WIFI_IF_STA, &getprotocol);
if (errwf != ESP_OK) {
	printf("Could not get protocol!");
}
if (getprotocol & WIFI_PROTOCOL_11N) {
	printf("Wi-Fi Protocol: 802.11n\n");
}
if (getprotocol & WIFI_PROTOCOL_11G) {
	printf("Wi-Fi Protocol: 802.11g\n");
}
if (getprotocol & WIFI_PROTOCOL_11B) {
	printf("Wi-Fi Protocol: 802.11b\n");
}
```

We can check which of the protocols are working. The TCP connection is handled by the program `tcp_client_v4.c`. Here, the client establishes connection with the TCP server, sends the image and awaits for the server response. This is done repeatedly to average the execution times of multiple runs, and then the connection is closed and the average time is printed to the serial monitor. Therefore, for this case, you need to open the serial monitor.
