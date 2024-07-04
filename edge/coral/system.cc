#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

// network
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/api.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/dns.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/sockets.h"

#include "libs/base/led.h"
#include "libs/tensorflow/utils.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <functional>

#include "include/micro_model_runner.h"
#include "libs/base/filesystem.h"
#include "libs/base/network.h"

#include "include/image.h"
#include "include/utils.h"

namespace coralmicro {
namespace {
constexpr int kTensorArenaSize = 560 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  vTaskDelay(pdMS_TO_TICKS(7000));

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();

  if (!tpu_context) {
    MicroPrintf("ERROR: Failed to get EdgeTpu context");
  }

  EnableWiFi();
  // NOTE: substitute with appropriate IP address and port number.
  int sockfd = SocketClient("172.16.0.227", 12345);
  int flag = 1;
  int result =
      setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag, sizeof(int));
  if (result < 0) {
    MicroPrintf("Couldn't disable Nagles algorithm");
  }
  if (sockfd < 0) {
    MicroPrintf("Connection Failed.\r\n");
    vTaskSuspend(nullptr);
  }

  std::vector<uint8_t> data;
  // Read one image and send over a socket.
  if (!coralmicro::LfsReadFile("image.bin", &data)) {
    while (1) {
      MicroPrintf("no such file");
    }
  }

  while (true) {
    uint64_t start_time = coralmicro::TimerMicros();

    if (WriteArray(sockfd, data.data(), data.size()) != IOStatus::kOk) {
      MicroPrintf("Failed to write sent request\r\n");
      vTaskSuspend(nullptr);
    }

    while (true) {
      if (int available = SocketAvailable(sockfd); available > 0) {
        std::vector<uint8_t> arr(available);
        if (ReadArray(sockfd, arr.data(), arr.size()) == IOStatus::kOk) {
          break;
        } else {
          break;
        }
      }
    }

    uint32_t send_time =
        static_cast<uint32_t>(coralmicro::TimerMicros() - start_time);
    MicroPrintf("%d", send_time);
  }
}
} // namespace
} // namespace coralmicro

extern "C" void app_main(void *param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
