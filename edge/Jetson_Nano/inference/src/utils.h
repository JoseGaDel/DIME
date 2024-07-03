#pragma once
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h" 
#include <cuda.h>



// Utility methods
namespace Util {
inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

std::vector<std::string> getFilesInDirectory(const std::string &dirPath) {
    std::vector<std::string> filepaths;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path().string());
    }
    return filepaths;
}
} // namespace Util


inline const char* severityString(nvinfer1::ILogger::Severity severity) {
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case nvinfer1::ILogger::Severity::kERROR:          return "ERROR";
        case nvinfer1::ILogger::Severity::kWARNING:        return "WARNING";
        case nvinfer1::ILogger::Severity::kINFO:           return "INFO";
        case nvinfer1::ILogger::Severity::kVERBOSE:        return "VERBOSE";
        default:                                           return "UNKNOWN";
    }
}

class TRTLogger : public nvinfer1::ILogger {
    public:
        virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
            //Prints warning messages with colors to terminal
            if (severity <= Severity::kINFO) {
              if (severity == Severity::kWARNING) {
                std::cout << "\033[1;33m" << "TRT Warning: " << msg << "\033[0m" << std::endl;
              } else if (severity == Severity::kERROR) {
                std::cout << "\033[1;31m" << "TRT Error: " << msg << "\033[0m" << std::endl;
              } else {
                std::cout << "TRT Info: " << msg << std::endl;
              }
               
            }
        }
} gLogger;


enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Options for the network
struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset
    // directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8
    // inference. Should be set to as large a batch number as your GPU will
    // support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // GPU device index
    int deviceIndex = 0;

    // preprocessing parameters
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};
    bool normalize = true;
    bool preprocess = true;
};


struct Image {
    uint16_t label;
    std::vector<uint8_t> data;
};



std::vector<Image> LoadCifar10(const std::string& filename, const int image_size = 3072,
                                int num_images_to_fetch = 10000) {
    /*
    Function to load the CIFAR-10 dataset from a binary file. The function reads 
    the images and labels and stores them in a vector of Image structures. The function
    assumes that the file is in the binary format used by the CIFAR-10 as explained 
    in https://www.cs.toronto.edu/~kriz/cifar.html.

    Inputs:
        - filename: A string containing the path to the binary file containing the dataset.
        - image_size: An integer specifying the size of the image in bytes. Default is 32x32x3 = 3072 bytes.
        - num_images_to_fetch: An integer specifying the number of images to fetch from the file. Default is 10000.
    
    Outpus:
        - A vector of Image structures containing the images and labels.

    The function uses the image_size parameter to calculate the number of images in the file. The function
    */

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Check file size
    file.seekg(0, std::ios::end);
    int file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of images in the file
    int num_images = file_size / (image_size + 1); // +1 byte for the label

    if (num_images_to_fetch > num_images) {
        throw std::runtime_error("Number of images to fetch exceeds the number of images in the file.");
    }

    std::vector<Image> images(num_images_to_fetch);
    for (int i = 0; i < num_images_to_fetch; ++i) {
        images[i].data.resize(image_size);
        // Store the image label
        file.read(reinterpret_cast<char*>(&images[i].label), 1);

        // Read the 3 RGB channels
        for (int j = 0; j < image_size; ++j) {
            file.read(reinterpret_cast<char*>(&images[i].data[j]), 1);
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


void copyImageToInputBufferNCHW(std::vector<uint8_t>& image_data, float* inputBuffer, 
                                int height, int width, int channels) {
    /*
    Function to copy an image to an input buffer in NCHW format. The image data is assumed to be in NHWC format and is
    copied without any preprocessing. TensorRT requires the buffer to be in NCHW format and the image is copied channel-wise
    in such format.
    • Inputs:
        - image_data: A vector containing the pixel values of the image.
        - inputBuffer: A pointer to the input buffer that will be used to feed the model.
        - height: The height of the image.
        - width: The width of the image.
        - channels: The number of channels in the image.
    • Outputs:
        - None. The image is copied directly to the input buffer.
    */

    // Assuming HxW image and inputBuffer is already in NCHW format

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                // Calculate indices for planar and interleaved formats
                int planarIndex = c * height * width + h * width + w;
                int nhwcIndex = (h * width + w) * channels + c;

                // Copy pixel value
                inputBuffer[nhwcIndex] = static_cast<float>(image_data[planarIndex]);
            }
        }
    }
};


void preprocessNCHW(std::vector<uint8_t>& image_data, float* inputBuffer, 
                    const std::vector<float>& mean, const std::vector<float>& std,
                    int height, int width, int channels) {
    /*
    Function to copy an image to an input buffer in NCHW format applying mean subtraction and std normalization. The image 
    data is assumed to be in NHWC and the preprocessing is applied channel-wise. The input buffer stored in NCHW format
    as required by TensorRT.
    • Inputs:
        - image_data: A vector containing the pixel values of the image.
        - inputBuffer: A pointer to the input buffer that will be used to feed the model.
        - mean: A vector containing the mean values for each channel.
        - std: A vector containing the standard deviation values for each channel.
    • Outputs:
        - None. The image is copied directly to the input buffer after applying the preprocessing steps.
    */

    // Assuming 32x32 image, inputBuffer in NCHW format, and mean/std have 3 elements
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                // Calculate indices for planar and interleaved formats
                int planarIndex = c * height * width + h * width + w;
                int nhwcIndex = (h * width + w) * channels + c; 

                // Preprocess and copy
                inputBuffer[nhwcIndex] = (static_cast<float>(image_data[planarIndex]) - mean[c]) / std[c];
            }
        }
    }
}


void preprocessNormalizeNCHW(std::vector<uint8_t>& image_data, float* inputBuffer, 
                                const std::vector<float>& mean, const std::vector<float>& std,
                                int height, int width, int channels) {
    /*
    Function to copy an image to an input buffer in NCHW format applying pixel normalization, mean subtraction and std normalization. The image 
    data is assumed to be in NHWC and the preprocessing is applied channel-wise. The input buffer is assumed to be in NCHW format.
    • Inputs:
        - image_data: A vector containing the pixel values of the image.
        - inputBuffer: A pointer to the input buffer that will be used to feed the model.
        - mean: A vector containing the mean values for each channel.
        - std: A vector containing the standard deviation values for each channel.
    • Outputs:
        - None. The image is copied directly to the input buffer after applying the preprocessing steps.
    */

    // Assuming 32x32 image, inputBuffer in NCHW format, and mean/std have 3 elements
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                // Calculate indices for planar and interleaved formats
                int planarIndex = c * height * width + h * width + w;
                int nhwcIndex = (h * width + w) * channels + c;

                // Preprocess and copy
                inputBuffer[nhwcIndex] = (static_cast<float>(image_data[planarIndex]) / 255.0f - mean[c]) / std[c];
            }
        }
    }
}


void preprocessImage(const std::vector<uint8_t>& image_data, float* inputBuffer, int width, int height, int channels) {
    // Define the mean and std for each channel
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};

    // Calculate the total number of pixels per channel
    int totalPixels = width * height;

    // Assuming image_data size is width * height * channels
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int srcIndex = h * width * channels + w * channels + c; // Index in NHWC layout
                int dstIndex = c * totalPixels + h * width + w; // Index in NCHW layout

                // Normalize the pixel value to [0,1]
                float normalized = image_data[srcIndex] / 255.0f;

                // Standardize the pixel value
                float standardized = (normalized - mean[c]) / std[c];

                // Store the result in the input buffer
                inputBuffer[dstIndex] = standardized;
            }
        }
    }
}
