#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <numeric>

#define SERVER_IP "172.16.0.241"    // Change this to the server IP address
#define SERVER_PORT 12345           // Change this to the server port number
#define DATA_FILE "../data/test_batch.bin" // Path to the file with the images to send

const int width = 32, height = 32, channels = 3;  // CIFAR-10 image dimensions
const int image_size = width * height * channels;


struct Image {
  std::vector<uint8_t> data;
  uint8_t label;
};

std::vector<Image> LoadCifar10(const std::string& filename) {
/*
    Utility to load the CIFAR-10 dataset from a binary file.
    - Input: filename: Path to the binary file
    - Output: images: A vector of images, where each image is represented as a struct with the following fields:
        - data: A vector of 3072 bytes (1024 bytes for each RGB channel)
        - label: An integer representing the image label

    Note: In this case we send the native image as raw 8-bit unsigned integers.
    Your model may need to store the image as floats instead of integers, which
    for this repo will be done in each particular case. This function may appear
    in other folders with different implementations.
*/

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Check file size
  file.seekg(0, std::ios::end);
  int file_size = file.tellg();
  if (file_size != 30730000) {
    throw std::runtime_error("Invalid file size: " + filename);
  }

  file.seekg(0, std::ios::beg);

  std::vector<Image> images(10000);
  for (int i = 0; i < 10000; ++i) {
    images[i].data.resize(3072);
    // Store the image label
    file.read(reinterpret_cast<char*>(&images[i].label), 1);

    // Read the 3 RGB channels
    for (int j = 0; j < 3072; ++j) {
      unsigned char value;
      file.read(reinterpret_cast<char*>(&value), 1);
      images[i].data[j] = static_cast<uint8_t>(value);
    }
  }

  return images;
};


std::vector<Image> LoadCifar10(const std::string& filename, int num_images, int width, int height, int channels) {
    /*
    Utility to load the CIFAR-10 dataset from a binary file.
    • Input: 
        - filename: Path to the binary file
        - num_images: Number of images to load
        - width: Width of the images
        - height: Height of the images
    • Output: images: A vector of images, where each image is represented as a struct with the following fields:
        - data: A vector of width x height x channels bytes (width x height bytes for each RGB channel)
        - label: An integer representing the image label

    Note: In this case we send the native image as raw 8-bit unsigned integers.
    Your model may need to store the image as floats instead of integers, which
    for this repo will be done in each particular case. This function may appear
    in other folders with different implementations.
*/
    const int image_size = width * height * channels;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);


    std::vector<Image> images(num_images);

    for (int i = 0; i < num_images; ++i) {
        images[i].data.resize(image_size);
        // Store the image label
        file.read(reinterpret_cast<char*>(&images[i].label), 1);

        // Read the image data
        for (int j = 0; j < image_size; ++j) {
            unsigned char value;
            file.read(reinterpret_cast<char*>(&value), 1);
            images[i].data[j] = static_cast<uint8_t>(value);
        }
    }

    return images;
};




int main(int argc, char* argv[]) {
    int client_fd;
    struct sockaddr_in server_addr;
    int response;

    // Get the size of the selected data file
    std::ifstream file(DATA_FILE, std::ios::binary);
    file.seekg(0, std::ios::end);
    int file_size = file.tellg();
    file.close();

    int num_images = file_size / (image_size + 1);  // +1 for the label

    // Determine the number of iterations. If the user doesn't choose a number of images,
    // by default we take the entire test set
    int num_iterations = num_images;
    if (argc > 1)
        num_iterations = std::min(num_iterations, std::stoi(argv[1]));


    // We load the test set to send the images to the server
    std::vector<Image> images = LoadCifar10(DATA_FILE, num_iterations, width, height, channels);

    // To measure the latency and transfer speeds
    std::vector<double> latencies;
    std::vector<double> speeds;

    // Create a socket
    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Define the server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    // Connect to the server
    if (connect(client_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("Connect failed");
        close(client_fd);
        exit(EXIT_FAILURE);
    }

    // The TCP connection will remain open for the entire duration of the test to
    // avoid the overhead of establishing a new connection for each image. The server
    // should be able to keep in the order of tens of thousands of open connections.

    for (int i = 0; i < num_iterations; ++i) {
        const Image& image = images[i];
        
        // Start the timer
        auto start = std::chrono::high_resolution_clock::now();

        // Send the image data to the server
        if (send(client_fd, image.data.data(), image.data.size(), 0) == -1) {
            perror("Send failed");
            close(client_fd);
            return 1;
        }

        // Receive the response from the server
        if (recv(client_fd, &response, sizeof(response), 0) == -1) {
            perror("Receive failed");
            close(client_fd);
            return 1;
        }

        // Measure and accumulate the latency and transfer speed
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = end - start;
        latencies.push_back(latency.count());

        double speed = image.data.size() / latency.count() * 8 / 1000;  // Convert to Mbps;
        speeds.push_back(speed);
    }

    double average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double average_speed = std::accumulate(speeds.begin(), speeds.end(), 0.0) / speeds.size();

    std::cout << "Average latency:        " << average_latency << " ms\n";
    std::cout << "Average transfer speed: " << average_speed << " Mbps\n";

    // Close the client socket
    close(client_fd);

    return 0;
}