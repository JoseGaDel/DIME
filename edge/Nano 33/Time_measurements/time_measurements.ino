/*
This program has been designed to run on the Arduino Nano 33 BLE Sense. It receives images via USB connection and runs
a TinyML model and a LR module. This program returns the computational time required for certain tasks. For comments on specific lines of code,
please check "main_program".
*/

#include <TensorFlowLite.h>
#include <string.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "quant_model.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "LR.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
const int kTensorArenaSize =  80 *1024;
const int imageSize = 3072;
uint8_t tensor_arena[kTensorArenaSize];
const int rows = 32;
const int columns = 32;
const int channels = 3;
int8_t image[rows][columns][channels];
const int numImages =  25;
byte imageData[numImages][imageSize];
int8_t off_array[numImages];
int imageIndex;
int dataIndex;
unsigned long programtime;
unsigned long infertime;
unsigned long lrtime;

}  

int findMaxIndex(int8_t arr[], int size) {
  int maxIndex = 0; // Assume the first element is the maximum

  for (int i = 1; i < size; ++i) {
    if (arr[i] > arr[maxIndex]) {
      // Update maxIndex if a larger value is found
      maxIndex = i;
    }
  }
  return maxIndex;
}


void setup() {

  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  model = tflite::GetModel(quant_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<9> micro_op_resolver; 
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddDetectionPostprocess();
  micro_op_resolver.AddReshape();


  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}


void loop() {  

  imageIndex = 0;
  dataIndex = 0;  

  while(imageIndex < numImages){
    if (Serial.available() > 0) {
      imageData[imageIndex][dataIndex] = Serial.read();
      dataIndex++;
    
      if (dataIndex >= imageSize) {
        dataIndex = 0;
        imageIndex++;
      }
    }
  }   
   
  delay(100);

  for(int8_t w = 0; w < numImages; w++){ 
    
    unsigned long start = micros();
    
    int index = 0;
    for (int j = 0; j < rows; j++){
      for (int i = 0; i < columns; i++){
        for (int c = 0; c < 3; c++){
          int value =  imageData[w][index] - 128;
          image[j][i][c] = value;
          index++;
        }
      }
    }

    memcpy(input->data.int8, image, sizeof(image));

    unsigned long startinf = micros();

    TfLiteStatus invoke_status = interpreter->Invoke();

    unsigned long endinf = micros();

    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }

    TfLiteTensor* output = interpreter->output(0);

    int8_t airplane_score = output->data.uint8[0]; 
    int8_t automobile_score = output->data.uint8[1];
    int8_t bird_score = output->data.uint8[2];
    int8_t cat_score = output->data.uint8[3];
    int8_t deer_score = output->data.uint8[4];
    int8_t dog_score = output->data.uint8[5];
    int8_t frog_score = output->data.uint8[6];
    int8_t horse_score = output->data.uint8[7];
    int8_t ship_score = output->data.uint8[8];
    int8_t truck_score = output->data.uint8[9];

    int8_t scores[] = {airplane_score,automobile_score,bird_score,cat_score,deer_score,dog_score,frog_score,horse_score,ship_score,truck_score};
    
    int8_t clase = findMaxIndex(scores, 10);
    
    unsigned long startlr = micros();

    off_array[w] = LRpredict(scores);

    unsigned long end = micros();
    

    programtime = end - start;
    infertime = endinf - startinf;
    lrtime = end - startlr;
    

    Serial.print("[");
    Serial.print(String(programtime));
    Serial.print(",");
    Serial.print(String(infertime));
    Serial.print(",");
    Serial.print(String(lrtime));
    Serial.print("]");

  }
  memset(image, 0, sizeof(image));
  memset(imageData, 0, sizeof(imageData));
  delay(100);
}