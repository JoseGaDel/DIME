#include <cstdio>
#include <vector>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "include/utils.h"

#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32

#define IMAGENET_IMAGE_WIDTH 224
#define IMAGENET_IMAGE_HEIGHT 224

static float prepare_pixel_resnet56(uint8_t pixel, float mean, float std) {
    return (static_cast<float>(pixel) / 255.0 - mean) / std;
}

static float prepare_pixel_alexnet(uint8_t pixel, float mean, float std) {
    return (static_cast<float>(pixel) - mean) / std;
}

static float prepare_pixel_imagenet(uint8_t pixel, float mean, float std) {
    return (static_cast<float>(pixel) / 255.0 - mean) / std;
}

std::vector<uint8_t> preprocess_cifar_image(std::vector<uint8_t> data, float scale, float zero_point) {
    std::vector<uint8_t> red(
        data.begin(), data.begin() + CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT);
    std::vector<uint8_t> green(
        data.begin() + CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT,
        data.begin() + 2 * CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT);
    std::vector<uint8_t> blue(
        data.begin() + 2 * CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT,
        data.end());

    std::vector<uint8_t> image;
    image.reserve(CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT * 3);

    for (int i = 0; i < CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT; ++i) {
#if defined(RESNET8)
        image.push_back(red[i]);
        image.push_back(green[i]);
        image.push_back(blue[i]);
#elif defined(RESNET56)
        float r = prepare_pixel_resnet56(red[i], 0.4914, 0.2023);
        float g = prepare_pixel_resnet56(green[i], 0.4822, 0.1994);
        float b = prepare_pixel_resnet56(blue[i], 0.4465,  0.2010);
        image.push_back(QuantizeFloatToUInt8(r, scale, zero_point));
        image.push_back(QuantizeFloatToUInt8(g, scale, zero_point));
        image.push_back(QuantizeFloatToUInt8(b, scale, zero_point));
#elif defined(ALEXNET)
        float r = prepare_pixel_alexnet(red[i], 125.307, 62.9932);
        float g = prepare_pixel_alexnet(green[i], 122.95, 62.0887);
        float b = prepare_pixel_alexnet(blue[i], 113.865,  66.7048);
        image.push_back(QuantizeFloatToUInt8(r, scale, zero_point));
        image.push_back(QuantizeFloatToUInt8(g, scale, zero_point));
        image.push_back(QuantizeFloatToUInt8(b, scale, zero_point));
#endif
    }

    return image;
}

std::vector<uint8_t> preprocess_imagenet_image(std::vector<uint8_t> data, float scale, float zero_point) {
    std::vector<uint8_t> red(
            data.begin(), data.begin() + IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT);
    std::vector<uint8_t> green(
            data.begin() + IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT,
            data.begin() + 2 * IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT);
    std::vector<uint8_t> blue(
            data.begin() + 2 * IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT,
            data.end());

    std::vector<uint8_t> image;
    image.reserve(IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT * 3);

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT; ++i) {
            if (c == 0) {
                float r = prepare_pixel_imagenet(red[i], 0.485, 0.229);
                image.push_back(QuantizeFloatToUInt8(r, scale, zero_point));
            }
            else if (c == 1) {
                float g = prepare_pixel_imagenet(green[i], 0.456, 0.224);
                image.push_back(QuantizeFloatToUInt8(g, scale, zero_point));
            }
            else {
                float b = prepare_pixel_imagenet(blue[i], 0.406,  0.225);
                image.push_back(QuantizeFloatToUInt8(b, scale, zero_point));
            }
        }
    }

    return image;
}

std::vector<uint8_t> preprocess_imagenet_image_uint8(uint8_t* data, float scale, float zero_point) {
    uint8_t* red = data;
    uint8_t* green = data + IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT;
    uint8_t* blue = data + 2 * IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT;

    std::vector<uint8_t> image;
    image.reserve(IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT * 3);

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < IMAGENET_IMAGE_WIDTH * IMAGENET_IMAGE_HEIGHT; ++i) {
            if (c == 0) {
                float r = prepare_pixel_imagenet(red[i], 0.485, 0.229);
                image.push_back(QuantizeFloatToUInt8(r, scale, zero_point));
            }
            else if (c == 1) {
                float g = prepare_pixel_imagenet(green[i], 0.456, 0.224);
                image.push_back(QuantizeFloatToUInt8(g, scale, zero_point));
            }
            else {
                float b = prepare_pixel_imagenet(blue[i], 0.406,  0.225);
                image.push_back(QuantizeFloatToUInt8(b, scale, zero_point));
            }
        }
    }

    return image;
}

