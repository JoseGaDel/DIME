#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/api.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/dns.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/sockets.h"

#include "libs/base/wifi.h"
#include <cmath>

void EnableWiFi() {
    bool success = coralmicro::WiFiTurnOn(true);
    if (!success) {
        printf("Failed to turn on Wi-Fi\r\n");
        vTaskSuspend(nullptr);
    } else {
        printf("Successfully turned Wi-Fi on\r\n");
    }
    success = coralmicro::WiFiConnect();
    if (!success) {
        printf("Failed to connect to Wi-Fi\r\n");
        vTaskSuspend(nullptr);
    }
    printf("Wi-Fi connected\r\n");
    auto our_ip_addr = coralmicro::WiFiGetIp();

    if (our_ip_addr.has_value()) {
        printf("DHCP succeeded, our IP is %s.\r\n", our_ip_addr.value().c_str());
    } else {
        printf("We didn't get an IP via DHCP, not progressing further.\r\n");
        vTaskSuspend(nullptr);
    }
}

float DequantizeInt8ToFloat(int8_t value, float scale, int zero_point) {
    return static_cast<float>(value - zero_point) * scale;
}

float DequantizeUInt8ToFloat(uint8_t value, float scale, int zero_point) {
    return static_cast<float>(value - zero_point) * scale;
}

uint8_t QuantizeFloatToUInt8(float value, float scale, int zero_point) {
    return static_cast<uint8_t>(std::round(value / scale) + zero_point);
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> result;
    float sum_exp = 0.0;

    for (float val : input) {
        sum_exp += std::exp(val);
    }

    for (float val : input) {
        result.push_back(std::exp(val) / sum_exp);
    }

    return result;
}

