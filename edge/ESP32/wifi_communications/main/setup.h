#ifndef SETUP
#define SETUP

// Credentials of the WiFi Access Point
#define SSID "jose-hotspot"
#define PASS ""  // leave empty if the wifi is open. If it has password, uncomment the desired authentication mode
#define  AUTHMODE WIFI_AUTH_WPA2_PSK
//#define AUTHMODE WIFI_AUTH_WEP
//#define AUTHMODE WIFI_AUTH_WPA_PSK
//#define AUTHMODE WIFI_AUTH_WPA2_PSK
//#define AUTHMODE WIFI_AUTH_WPA3_PSK
//#define AUTHMODE WIFI_AUTH_WPA2_WPA3_PSK
//#define AUTHMODE WIFI_AUTH_WAPI_PSK


// The WiFi Access Point can be unstable. Change this property to make multiple
// attempts at connecting WiFi AP if you encounter trouble with connection
#define MAX_FAILURES 10

// You can choose which WiFi protocols to include in the configuration:
#define WIFI_PROTOCOLS (WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N)

// IP address where the TCP server is hosted and port it is listening to
#define HOST_IP_ADDR "10.42.0.1"
#define PORT 12345

// Dimensions of the image array
#define ROWS 32
#define COLUMNS 32
#define CHANNELS 3

// number of times to offload to take the average transfer time
#define MEASUREMENTS 300

#endif
