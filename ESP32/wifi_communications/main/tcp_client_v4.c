#include "sdkconfig.h"
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>
#include <netdb.h>            // struct addrinfo
#include <arpa/inet.h>
#include "esp_netif.h"
#include "esp_log.h"
//#include "image.h"
#include "setup.h"         // all the macros are stored here to more easely change the set up
#include "timer_u32.h"     // For accurate timing. Source: https://github.com/OliviliK/ESP32_timer_u32.git
#if defined(CONFIG_EXAMPLE_SOCKET_IP_INPUT_STDIN)
#include "addr_from_stdin.h"
#endif

static const char *TAG = "TCP_CLIENT";
// For timing
uint32_t dt,t0;


void tcp_client(void)
{
/*
    Offloading function. Establishes communication with the server, sends the image via TCP
    for inference and receives a single byte back with the inference class. For testing proposes
    the image data is made internally but could be modified to accept it as input. The function
    does not return the class received from the server but in production can be modified to
    return such value.
*/
    // We store the image on the heap because in the stack we can have overflows
    uint8_t *image = (uint8_t *)malloc(ROWS * COLUMNS * CHANNELS);
    for (int i = 0; i < (ROWS * COLUMNS * CHANNELS); i++) {
        image[i] = (uint8_t)(i % 256);
        //printf("%d ", image[i]);
    }

    // Initial setup
    char rx_buffer[128];
    char host_ip[] = HOST_IP_ADDR;
    int addr_family = 0;
    int ip_protocol = 0;
    int iterations = 0;
    float elapsed_time = 0;
    int sock;

#if defined(CONFIG_EXAMPLE_IPV4)
    struct sockaddr_in dest_addr;
    inet_pton(AF_INET, host_ip, &dest_addr.sin_addr);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);
    addr_family = AF_INET;
    ip_protocol = IPPROTO_IP;
#elif defined(CONFIG_EXAMPLE_SOCKET_IP_INPUT_STDIN)
    struct sockaddr_storage dest_addr = { 0 };
    ESP_ERROR_CHECK(get_addr_from_stdin(PORT, SOCK_STREAM, &ip_protocol, &addr_family, &dest_addr));
#endif
    while(1) {
        // create socket for communication
        sock =  socket(addr_family, SOCK_STREAM, ip_protocol);
        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            continue;
        }
        ESP_LOGI(TAG, "Socket created, connecting to %s:%d", host_ip, PORT);

        // connect the created socket
        int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (err != 0) {
            ESP_LOGE(TAG, "Socket unable to connect: errno %d", errno);
            continue;
        } else {
            ESP_LOGI(TAG, "Successfully connected");
            break;
        }
    }

    // we offload `MEASUREMENTS` times (defined in setup.h) to average the transfer time
    while (iterations < MEASUREMENTS) {

        // start measuring time. 
        t0 = timer_u32();
        while (1) {
            // send to the server the length of the message so it knows when to stop reading from buffer
            int err = send(sock, image, (ROWS * COLUMNS * CHANNELS), 0);
 
            if (err < 0) {
                ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
                break;
            }

            int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);

            // Error occurred during receiving
            if (len < 0) {
                ESP_LOGE(TAG, "recv failed: errno %d", errno);
                break;
            }
            // Data received
            else {
                rx_buffer[len] = 0; // Null-terminate whatever we received and treat like a string
                ESP_LOGI(TAG, "Received %d bytes from %s:", len, host_ip);
                ESP_LOGI(TAG, "%s", rx_buffer);
                iterations++;
                break;
            }
        }
        // take measurement of elapsed time
        dt = timer_u32() - t0;
        elapsed_time += timer_delta_ms(dt);
    }

    // if communication was terminated, close the socket
    if (sock != -1) {
        ESP_LOGE(TAG, "End of communication. Shutting down socket.");
        shutdown(sock, 0);
        close(sock);
    }

    printf("Elapsed time: %f ms\n", elapsed_time/MEASUREMENTS); 
}
