void EnableWiFi();
float DequantizeInt8ToFloat(int8_t value, float scale, int zero_point);
float DequantizeUInt8ToFloat(uint8_t value, float scale, int zero_point);
uint8_t QuantizeFloatToUInt8(float value, float scale, int zero_point);
std::vector<float> softmax(const std::vector<float> &input);
