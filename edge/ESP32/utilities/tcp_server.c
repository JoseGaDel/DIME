#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/tcp.h> // For TCP_NODELAY


#define SERVER_PORT 12345

#define ROWS 32
#define COLUMNS 32
#define CHANNELS 3
#define BUFFER_SIZE (ROWS * COLUMNS * CHANNELS)

int inference(uint8_t *image) {
    // Placeholder for TinyML inference
    // Return an integer between 0 and 9 to represent the class
    return rand() % 10; // Random number for demonstration
}


int main() {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size;
    uint8_t buffer[BUFFER_SIZE];
    int received_bytes, response;

    // Create a socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Disable Nagle's algorithm to prevent the outgoing packet to be buffered as is very small
    // This can help minimize the latency
    int flag = 1;
    if (setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    // Define the server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    // Bind the socket to the server address
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_fd, 5) == -1) {
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on port %d...\n", SERVER_PORT);
    // Accept incoming connections
    client_addr_size = sizeof(client_addr);
    client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_size);
    if (client_fd == -1) {
        perror("Accept failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    printf("Connection accepted from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    while (1) {

        // If the incoming data is too big to fit in one packet, we have to read iteratively from buffer until
        // the message is complete. For that, we tally the total size of the message received in the following loop
        // until it matches the expected size of BUFFER_SIZE:
        int total_bytes_received = 0;
        while (total_bytes_received < BUFFER_SIZE) {

            int bytes_received = recv(client_fd, buffer + total_bytes_received, 
                                    BUFFER_SIZE - total_bytes_received, 0);
            if (bytes_received <= 0)
            {   
                if (bytes_received == 0) {
                    printf("Connection terminated, restarting socket\n");
                } else {
                    printf("Failed to receive, restarting socket\n");
                }
                close(client_fd); 
                // Accept a new connection
                client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_size);
                if (client_fd == -1) {
                    perror("Accept failed");
                    close(server_fd);
                    exit(EXIT_FAILURE); 
                } else {
                    printf("Connection accepted from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
                }
                total_bytes_received = 0;
                continue;
            }
            total_bytes_received += bytes_received; // accumulate the received bytes to the total
        }
        // 
        //printf("Received data\n");
        /*
        _________  PERFORM INFERENCE  __________    
        */

        // Process the image and get a response
        //response = inference(buffer);
        response = 0;

        // Send the response
        char response_char = (char)response + '0'; // Convert to ASCII character
        if (send(client_fd, &response_char, sizeof(response_char), 0) == -1) {
            perror("Send failed");
            // ... error handling
        }

        //printf("Response sent: %d\n", response);

    }
    // Close the server socket
    close(server_fd);

    return 0;
}
