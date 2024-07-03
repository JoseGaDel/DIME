#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/tcp.h> // TCP_NODELAY
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <numeric>
#include <sstream>
#include <iomanip> // std::setw and std::setfill
#include <cmath>

#define SERVER_IP "0.0.0.0"    // Change this to the server IP address
#define SERVER_PORT 12345           // Change this to the server port number
#define DATA_PATH "../inference/data/" // Path to the file with the images to send


struct Image {
    std::vector<uint8_t> data;
    uint8_t label;
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


    std::vector<Image> images(1);

    for (int i = 0; i < 1; ++i) {
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


std::vector<uint16_t> load_labels(const std::string& label_file) {
    
    std::vector<uint16_t> labels;
    std::ifstream file(label_file);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + label_file);
    }

    std::string line;
    // Read lines from the file and store them in a vector
    while (std::getline(file, line)) {
        // Convert the string to an unsigned 16-bit integer
        uint16_t label;
        std::istringstream iss(line);
        if (!(iss >> label)) {
            std::cerr << "Error converting label to uint16_t: " << line << std::endl;
            continue;  // Skip invalid lines
        }
        labels.push_back(label);
    }

    file.close();

    return labels;
}


void load_bin(const std::string& filename, std::vector<uint8_t>& image) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // read binary data
    file.read(reinterpret_cast<char*>(image.data()), image.size());

    file.close();
}




int main(int argc, char* argv[]) {
    // take from command line wether to deal with imagenet or CIFAR-10
    // if CIFAR-10, the image size is 32x32x3
    // if ImageNet, the image size is 224x224x3
    // if CIFAR-10, the number of classes is 10
    // if ImageNet, the number of classes is 1000
    std::string dataset = argv[1];
    int width, height, channels;
    int image_size;
    const int num_iterations = 10000;
    std::vector <uint8_t> imageData;
    uint16_t response;

    // load a representative image from the selected dataset

    if (dataset == "CIFAR-10" || dataset == "cifar-10" || dataset == "cifar10" || dataset == "cifar") {
        width = 32, height = 32, channels = 3;  // CIFAR-10 image dimensions
        image_size = width * height * channels;

        std::ostringstream filename;
        filename << DATA_PATH << "test_batch.bin";

        std::vector<Image> images = LoadCifar10(filename.str(), 1, width, height, channels);
        imageData = images[0].data;
        images.clear();

    } else if (dataset == "ImageNet" || dataset == "ImageNet1k" || dataset == "ImageNet-1k" || dataset == "imagenet" || dataset == "imagenet" || dataset == "imagenet-1k" || dataset == "imagenet1k") {
        width = 224, height = 224, channels = 3;  // Imagenet image dimensions
        image_size = width * height * channels;

        imageData.resize(width * height * channels);

        std::ostringstream filename;
        filename << DATA_PATH << "bin/ILSVRC2012_val_" << std::setw(8) << std::setfill('0') << 0+1 << ".bin";
        load_bin(filename.str(), imageData);
    } else {
        std::cout << "Invalid dataset. Please choose between CIFAR-10 and ImageNet" << std::endl;
        return 1;
    }
    
    int client_fd;
    struct sockaddr_in server_addr;

    // Create a socket
    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Disable Nagle's algorithm
    int flag = 1;
    if (setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int)) == -1) {
        perror("TCP_NODELAY setsockopt failed");
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
#ifdef LATENCY
    // Measure the latency of the connection. Results will be stored in a file
    // called latencies.txt so we can get average and standard deviation

    double* latencies = (double*)malloc(sizeof(double) * num_iterations);

    for (int i = 0; i < num_iterations; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        // Send the image data to the server
        if (send(client_fd, imageData.data(), imageData.size(), 0) == -1) {
            perror("Send failed");
            close(client_fd);
            return 1;
        }

        // Receive response from the server
        int bytes_received = 0;
        while (bytes_received < sizeof(response)) {
            int result = recv(client_fd, reinterpret_cast<char*>(&response) + bytes_received, sizeof(response) - bytes_received, 0);
            if (result < 0) {
                std::cerr << "Receive failed" << std::endl;
                close(client_fd);
                return 1;
            }
            bytes_received += result;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
        latencies[i] = duration.count() / 1e6;
    }

    // calculate the average latency and speed
    double average_latency = std::accumulate(latencies, latencies + num_iterations, 0.0) / num_iterations;

    double average_speed = imageData.size() / average_latency * 8 / 1000;  // Convert to Mbps;

    // save latencies in file
    std::ofstream output_file;
    output_file.open("latencies.txt");
    for (int i = 0; i < num_iterations; ++i) {
        output_file << latencies[i] << std::endl;
    }
    output_file.close();
    std::cout << "Latencies saved in ../data/latencies.txt" << std::endl;


    std::cout << "Average latency:        " << average_latency << " ms\n";
    std::cout << "Average transfer speed: " << average_speed << " Mbps\n";
#endif

#ifdef ENERGY
// Measure the energy consumption of the connection. Run indefinitely to record power draw
// for as long as needed. No data is recorded.
    while (true) {
        // Send the image data to the server
        if (send(client_fd, imageData.data(), imageData.size(), 0) == -1) {
            perror("Send failed");
            close(client_fd);
            return 1;
        }

        // Receive response from the server
        int bytes_received = 0;
        while (bytes_received < sizeof(response)) {
            int result = recv(client_fd, reinterpret_cast<char*>(&response) + bytes_received, sizeof(response) - bytes_received, 0);
            if (result < 0) {
                std::cerr << "Receive failed" << std::endl;
                close(client_fd);
                return 1;
            }
            bytes_received += result;
        }
    } 
#endif         // Close the client socket
    close(client_fd);

    return 0;
}
