#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "libs/base/led.h"
#include "libs/tensorflow/utils.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <functional>

#include "include/logistic_regression.h"
#include "include/micro_model_runner.h"
#include "libs/base/filesystem.h"

#include "include/image.h"
#include "include/utils.h"

#include "include/cifar10_reader.h"

#define IMAGE_BATCH_SIZE 10000

#if defined(RESNET8)
constexpr char model_path[] = "models/resnet8_quant_edgetpu.tflite";
#elif defined(RESNET56)
constexpr char model_path[] = "models/resnet56_quant_edgetpu.tflite";
#elif defined(ALEXNET)
constexpr char model_path[] = "models/alexnet_quant_edgetpu.tflite";
#endif

namespace coralmicro {
namespace {
constexpr int kTensorArenaSize = 500 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();

  if (!tpu_context) {
    MicroPrintf("ERROR: Failed to get EdgeTpu context");
  }

  cifar::CIFAR10_dataset cifar_dataset;

  std::vector<uint8_t> model;

  if (!LfsReadFile(model_path, &model)) {
    MicroPrintf("ERROR: Failed to load model: %s\r\n", model_path);
    vTaskSuspend(nullptr);
  }

  cifar::read_file("datasets/cifar10/test_batch.bin", cifar_dataset,
                   IMAGE_BATCH_SIZE,
                   [] { return std::vector<uint8_t>(3 * 32 * 32); });

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
  MicroPrintf("image label,inference results,inference latency "
              "(microseconds),logistic regression,logistic regression latency "
              "(microseconds)\r\n");

  for (size_t j = 0; j < IMAGE_BATCH_SIZE; j++) {
    auto input_tensor = runner.get_input_tensor();

    std::vector<uint8_t> image =
        preprocess_cifar_image(cifar_dataset.images[j], runner.input_scale(),
                               runner.input_zero_point());

#if defined(RESNET8)
    for (size_t x = 0; x < cifar_dataset.images[j].size(); x++) {
      if (image[x] <= 127) {
        input_tensor->data.int8[x] = (static_cast<int8_t>(image[x])) - 128;
      } else {
        input_tensor->data.int8[x] = static_cast<int8_t>(image[x] - 128);
      }
    }
#elif defined(RESNET56) || defined(ALEXNET)
    std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image.data(),
                image.size());
#endif
    runner.Invoke(true);
    std::string result_string = "";
    result_string.append(std::to_string(cifar_dataset.labels[j]));
    result_string.append(",");

    std::vector<float> results;
    std::vector<float> softmax_results;

    for (size_t i = 0; i < 10; i++) {
      float converted;
#if defined(RESNET8)
      converted =
          DequantizeInt8ToFloat(runner.GetOutput(0)[i], runner.output_scale(),
                                runner.output_zero_point());
#elif defined(RESNET56) || defined(ALEXNET)
      converted =
          DequantizeUInt8ToFloat(runner.GetOutput(0)[i], runner.output_scale(),
                                 runner.output_zero_point());
#endif
      results.push_back(converted);
    }

#ifdef RESNET56
    softmax_results = softmax(results);
#endif

    uint64_t start_time = coralmicro::TimerMicros();
#if defined(RESNET8) || defined(ALEXNET)
    uint8_t prediction = LRpredict(results);
#elif defined(RESNET56)
    uint8_t prediction = LRpredict(softmax_results);
#endif
    uint32_t lr_time =
        static_cast<uint32_t>(coralmicro::TimerMicros() - start_time);

    for (size_t i = 0; i < 10; i++) {
#if defined(RESNET56)
      result_string.append(std::to_string(softmax_results[i]));
#elif defined(RESNET8) || defined(ALEXNET)
      result_string.append(std::to_string(results[i]));
#endif
      if (i < (10 - 1)) {
        result_string.append(";");
      } else {
        result_string.append(",");
      }
    }

    result_string.append(std::to_string(runner.get_inference_time()));
    result_string.append(",");
    result_string.append(std::to_string(prediction));
    result_string.append(",");
    result_string.append(std::to_string(lr_time));
    MicroPrintf("%s", result_string.c_str());
  }
}
} // namespace
} // namespace coralmicro

extern "C" void app_main(void *param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
