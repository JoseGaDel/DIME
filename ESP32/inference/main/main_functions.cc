/*
Program to perform inference on a 32x32x3 image using a CNN model trained on the CIFAR-10 dataset. The data
is fed to the program over serial communication. The program performs inference on the data and sends the 
results and the execution time back over serial communication. The results are sent as a string, with the first
10 values being the quantized output of the model, the 11th value being the logistic regression result, and the
12th to 14th values being the execution times of the inference, logistic regression and the entire process. When
the program is flashed, the serial monitor must not be opened, as it will interfere with the serial communication.

This program is based on the example program provided by Espressif for the ESP32, which can be found here:
    https://github.com/espressif/esp-tflite-micro
*/

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "LR.h"
#include "output_handler.h"

#include "esp_system.h"
#include "timer_u32.h"
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/uart.h>
#include <stdio.h>
#include <string>
//#include <sstream>

#define UART_PORT UART_NUM_0
#define BUF_SIZE 3072  // Adjust the buffer size as needed


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


constexpr int kTensorArenaSize = 55*1024;
uint8_t tensor_arena[kTensorArenaSize];

const int messageSize = 3072;
int8_t scores[10];
// Buffer for the final string
char dataString[200];

/*
Set baud rate for serial communication. It must be the same in the companion program that sends the data.
According to espressif:
"The baud rate is limited to 115200 when esptool establishes the initial connection, higher speeds are only used for data transfers.
Most hardware configurations will work with -b 230400, some with -b 460800, -b 921600 and/or -b 1500000 or higher."
Baud rates above 460800 are ill advised
*/

#define BAUD_RATE 460800

#define ROWS 32
#define COLUMNS 32
#define CHANNELS 3
int8_t image[ROWS][COLUMNS][CHANNELS];

// buffer to store received data
uint8_t rx_buffer[BUF_SIZE];

uint32_t dt, t0_total, t0_inference;

}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Configure UART for communication
  uart_config_t uart_config = {
      .baud_rate = BAUD_RATE,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .rx_flow_ctrl_thresh = 0,
  };
  uart_param_config(UART_PORT, &uart_config);
  uart_driver_install(UART_PORT, BUF_SIZE * 2, 0, 0, NULL, 0);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();
  resolver.AddDetectionPostprocess();
  resolver.AddReshape();
  // if (resolver.AddFullyConnected() != kTfLiteOk) {
  //   return;
  // }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Check if there is any data available on the UART
  int rx_length = uart_read_bytes(UART_PORT, rx_buffer, BUF_SIZE, portMAX_DELAY);

  // Record the start time of the entire process (preprocessing, inference and LR)
  t0_total = timer_u32();

  // If there is data available, store it in an array
  if (rx_length > 0) {
    int index = 0;
      // Store the 1-D received data into 3-D image[32][32][3]
    for (int c = 0; c < CHANNELS; c++) {
      for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
          image[i][j][c] = rx_buffer[index] - 128; // casting directly tu int8_t will give incorrect results
          //input->data.int8[index] = imageData[index] - 128;  //<- this could replace the memcpy below
          index++;
        }
      }
    }

    // *****************  INFERENCE ****************
    // presumably, if buffers are 4-byte aligned, memcpy() can copy 32 bits at a time and should be faster than the
    // line commented in the for loop above, which does the same: copy the image array into the input tensor.
    memcpy(input->data.int8, image, sizeof(image));

    // Record the start time of just the inference
    t0_inference = timer_u32();
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed\n");
      return;
    }

    TfLiteTensor* output = interpreter->output(0);

    // Record the end time of the inference
    dt = timer_u32() - t0_inference;
    float elapsed_time_inference = timer_delta_us(dt);

    // For sending data over serial communication, we have chosen to send it as a string. When we attempt to send it
    // as a byte array, the data is not received correctly on the other end. The string is comma separated, with the
    // first 10 values being the quantized output of the model, the 11th value being the logistic regression result,
    // and the 12th value being the execution time of the inference.
    dataString[0] = '\0';  // Initialize to an empty string

    // Temporary buffer for each number
    char tempBuffer[20];
    // int outputSize = 10;
    // Obtain the quantized output from model's output tensor and add it to the output string.
    for (int i = 0; i < 10; ++i) {
      scores[i] = output->data.int8[i];
      snprintf(tempBuffer, sizeof(tempBuffer), "%d,", scores[i]);
      strcat(dataString, tempBuffer);
    }

    // Perform logistic regression to estimate the validity of the inference.
    t0_inference = timer_u32();
    uint8_t log_reg = LRpredict(scores);
    // Record the end time. This is the same for both the LR alone and the entire process, so we can use just one measurement.
    dt = timer_u32();
    // Record the end time of the complete process
    float elapsed_time_LR = timer_delta_us(dt - t0_inference);
    float elapsed_time_total = timer_delta_us(dt - t0_total);

    // Add the Logistic Regression result to the output string
    snprintf(tempBuffer, sizeof(tempBuffer), "%d,", log_reg);
    strcat(dataString, tempBuffer);

    // Add all execution times to the output string
    snprintf(tempBuffer, sizeof(tempBuffer), "%f,", elapsed_time_inference);
    strcat(dataString, tempBuffer);
    snprintf(tempBuffer, sizeof(tempBuffer), "%f,", elapsed_time_LR);
    strcat(dataString, tempBuffer);
    snprintf(tempBuffer, sizeof(tempBuffer), "%f", elapsed_time_total);
    strcat(dataString, tempBuffer);

    // Send a signal to indicate that a message is about to be sent
    uart_write_bytes(UART_PORT, "#", 1);

    // Print the final string so that the receiving program can read it
    printf("%s\n", dataString);

  }
}
