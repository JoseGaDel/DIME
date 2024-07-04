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

#include "libs/base/filesystem.h"
#include "libs/base/network.h"
#include "include/micro_model_runner.h"
#include "include/logistic_regression.h"

#include "include/image.h"
#include "include/utils.h"

#define IMAGE_BATCH_SIZE 200
#define PORT 65432
#define BUFFER_SIZE 9408
#define IMAGE_SIZE 150528

#if defined(RESNET18)
constexpr char model_path[] = "models/resnet18_quant_edgetpu.tflite";
#elif defined(RESNET50)
constexpr char model_path[] = "models/resnet50_quant_edgetpu.tflite";
#endif

namespace coralmicro {
namespace {
constexpr int kTensorArenaSize = 500 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

size_t process_image(tflite::MicroModelRunner<int8_t, int8_t, 14> *runner,
                          uint8_t *buffer) {
  std::vector<float> results;
  auto input_tensor = runner->get_input_tensor();

  std::vector<uint8_t> image = preprocess_imagenet_image_uint8(
      buffer,
      runner->input_scale(), runner->input_zero_point());

  std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image.data(),
              image.size());

  runner->Invoke(true);

  for (size_t i = 0; i < 1000; i++) {
    float converted =
        DequantizeUInt8ToFloat(runner->GetOutput(0)[i], runner->output_scale(),
                               runner->output_zero_point());
    results.push_back(converted);
  }

  coralmicro::TimerInit();
  uint64_t start_time = coralmicro::TimerMicros();;
  uint8_t lr_output = LRpredict(results);
  uint32_t lr_time = static_cast<uint32_t>(coralmicro::TimerMicros() - start_time);

  size_t distance = std::distance(results.begin(),
                            std::max_element(results.begin(), results.end()));

  std::string result_string = "";
  result_string.append(std::to_string(distance));
  result_string.append(",");
  result_string.append(std::to_string(lr_output));
  result_string.append(",");
  result_string.append(std::to_string(runner->get_inference_time()));
  result_string.append(",");
  result_string.append(std::to_string(lr_time));

  MicroPrintf("%s", result_string.c_str());

  return distance;
}

void handle_connection(int new_socket, tflite::MicroModelRunner<int8_t, int8_t, 14>* runner) {
  while (1) {
    uint8_t *buffer = (uint8_t *)malloc(IMAGE_SIZE * sizeof(uint8_t));
    int total_bytes_received = 0;
    int bytes_received = 0;

    while (total_bytes_received < IMAGE_SIZE) {
      bytes_received = recv(new_socket, buffer + total_bytes_received,
                            IMAGE_SIZE - total_bytes_received, 0);
      if (bytes_received <= 0) {
        if (bytes_received < 0) {
          MicroPrintf("recv");
          vTaskSuspend(nullptr);
        } else {
          MicroPrintf("Connection closed by client.\n");
          vTaskSuspend(nullptr);
        }
      }

      total_bytes_received += bytes_received;
    }

    size_t result = process_image(runner, buffer);

    const char *ack = "ACK";
    lwip_send(new_socket, ack, strlen(ack), 0);

    free(buffer);
  }

  lwip_close(new_socket);
}

[[noreturn]] void Main() {
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();

  if (!tpu_context) {
    MicroPrintf("ERROR: Failed to get EdgeTpu context");
  }

  EnableWiFi();
  int sockfd = SocketServer(65432, 0);

  MicroPrintf("Server is listening on port %d", PORT);

  std::vector<uint8_t> model;

  if (!LfsReadFile(model_path, &model)) {
    MicroPrintf("ERROR: Failed to load model: %s\r\n", model_path);
    vTaskSuspend(nullptr);
  }

  tflite::MicroMutableOpResolver<14> resolver;
  resolver.AddDequantize();
  resolver.AddAdd();
  resolver.AddQuantize();
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddPad();
  resolver.AddMean();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();
  resolver.AddDetectionPostprocess();
  resolver.AddReshape();
  resolver.AddMul();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  static tflite::MicroModelRunner<int8_t, int8_t, 14> runner(
      model.data(), resolver, tensor_arena, kTensorArenaSize);

  // Accept incoming connections and handle them
  int new_socket;
  while ((new_socket = SocketAccept(sockfd)) >= 0) {
    MicroPrintf("Connection accepted\n");
    handle_connection(new_socket, &runner);
    MicroPrintf("Connection closed\n");
  }

  if (new_socket < 0) {
    MicroPrintf("accept");
    vTaskSuspend(nullptr);
  }
}
} // namespace
} // namespace coralmicro

extern "C" void app_main(void *param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
