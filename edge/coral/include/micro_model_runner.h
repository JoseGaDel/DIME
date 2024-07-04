/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <functional>

#include "third_party/tflite-micro/tensorflow/lite/micro/kernels/micro_ops.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "libs/base/timer.h"

namespace tflite {
template <typename inputT, typename outputT, int numOps>
class MicroModelRunner {
public:
  MicroModelRunner(const uint8_t *model,
                   MicroMutableOpResolver<numOps> &resolver,
                   uint8_t *tensor_arena, int tensor_arena_size)
      : model_(tflite::GetModel(model)), reporter_(&micro_reporter_),
        interpreter_(model_, resolver, tensor_arena, tensor_arena_size,
                     reporter_) {

    if (model_->version() != TFLITE_SCHEMA_VERSION) {
      printf("Model provided is schema version %ld not equal "
             "to supported version %d.",
             model_->version(), TFLITE_SCHEMA_VERSION);
      return;
    }
    if (interpreter_.AllocateTensors() != kTfLiteOk) {
      while (1) {
        printf("Couldn't allocate tensors for the model.\r\n");
      }
    }

    if (interpreter_.inputs().size() != 1) {
      while (1) {
        printf("ERROR: Model must have only one input tensor\r\n");
      }
      vTaskSuspend(nullptr);
    }
  }

  void Invoke(bool micros) {
    // Run the model on this input and make sure it succeeds.
    coralmicro::TimerInit();
    uint64_t start_time;
    if (micros) {
      start_time = coralmicro::TimerMicros();
    } else {
      start_time = coralmicro::TimerMillis();
    }
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (micros) {
      inference_time_ =
          static_cast<uint32_t>(coralmicro::TimerMicros() - start_time);
    } else {
      inference_time_ =
          static_cast<uint32_t>(coralmicro::TimerMillis() - start_time);
    }
    if (invoke_status != kTfLiteOk) {
      printf("Invoke failed.\r\n");
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

  void tflite_set_input(const void *data) {
    auto input = interpreter_.input(0);
    memcpy(input->data.int8, data, input->bytes);
  }

  void SetInput(const inputT *custom_input) {
    TfLiteTensor *input = interpreter_.input(0);
    inputT *input_buffer = tflite::GetTensorData<inputT>(input);

    int input_length = input->bytes / sizeof(inputT);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = custom_input[i];
    }
  }

  TfLiteTensor *get_input_tensor() { return interpreter_.input(0); }

  outputT *GetOutput(int i) {
    return tflite::GetTensorData<outputT>(interpreter_.output(i));
  }

  float *input_tensor() { return interpreter_.typed_input_tensor<float>(0); }

  void set_typed_input_tensor(float data[2]) {
    float *input_tensor = interpreter_.typed_input_tensor<float>(0);
    std::memcpy(input_tensor, data, sizeof(float) * 2);
  }

  float *get_typed_output() { return interpreter_.output(0)->data.f; }

  int get_typed_output_size() {
    return interpreter_.output(0)->bytes / sizeof(float);
  }

  int input_size() { return interpreter_.input(0)->bytes / sizeof(inputT); }

  int8_t *get_input() { return interpreter_.input(0)->data.int8; }

  int output_size() { return interpreter_.output(0)->bytes / sizeof(outputT); }
  int output_size_1() {
    return interpreter_.output(1)->bytes / sizeof(outputT);
  }

  size_t outputs_size() { return interpreter_.outputs_size(); }

  float output_scale() { return interpreter_.output(0)->params.scale; }

  int output_zero_point() { return interpreter_.output(0)->params.zero_point; }

  float input_scale() { return interpreter_.input(0)->params.scale; }

  int input_zero_point() { return interpreter_.input(0)->params.zero_point; }

  int arena_used_bytes() { return interpreter_.arena_used_bytes(); }

  unsigned long get_inference_time() { return inference_time_; }

private:
  const tflite::Model *model_;
  uint32_t inference_time_;
  tflite::MicroErrorReporter micro_reporter_;
  tflite::ErrorReporter *reporter_;

public:
  tflite::MicroInterpreter interpreter_;
};
} // namespace tflite
