float dequantize(int8_t value, float scale, int zero_point);

std::vector<uint8_t> preprocess_cifar_image(std::vector<uint8_t> data,
                                            float scale, float zero_point);
std::vector<uint8_t> preprocess_imagenet_image(std::vector<uint8_t> data,
                                               float scale, float zero_point);
std::vector<uint8_t> preprocess_imagenet_image_uint8(uint8_t *data, float scale,
                                                     float zero_point);
