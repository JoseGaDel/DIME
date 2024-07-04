#ifndef LOGISTIC_REGRESSION_HEADER_H
#define LOGISTIC_REGRESSION_HEADER_H

#include <algorithm>
#include <climits>
#include <cmath>

struct findHighestValues {
  void operator()(std::vector<float> arr, float size, float &firstHighest,
                  float &secondHighest);
};

#if defined(RESNET8)
const float weight1 = 5.93658349;
const float weight2 = -3.04197074;
const float beta = -4.63391351;
#elif defined(RESNET56)
const float weight1 = 5.21640018;
const float weight2 = -3.70736749;
const float beta = -4.695974;
#elif defined(ALEXNET)
const float weight1 = 5.39107836;
const float weight2 = -1.03445988;
const float beta = -4.33089237;
#elif defined(RESNET18)
const float weight1 = 4.95192416;
const float weight2 = -1.7800178;
const float beta = -3.02860354;
#elif defined(RESNET50)
const float weight1 = 5.11103057;
const float weight2 = -2.07690727;
const float beta = -3.45594814;
#endif

float value1;
float value2;

uint8_t LRpredict(std::vector<float> scores) {
#if defined(RESNET8) || defined(RESNET56) || defined(ALEXNET)
  findHighestValues()(scores, 10, value1, value2);
#elif defined(RESNET18) || defined(RESNET50)
  findHighestValues()(scores, 1000, value1, value2);
#endif

  float weighted_sum = beta + weight1 * value1 + weight2 * value2;
  float probability = 1.0 / (1.0 + exp(-weighted_sum));

  if (probability >= 0.5) {
    return 1; // Classify as positive
  } else {
    return 0; // Classify as negative
  }
}

void findHighestValues::operator()(std::vector<float> arr, float size,
                                   float &firstHighest, float &secondHighest) {
  // Initialize the highest and second highest values
  firstHighest = -1.0;
  secondHighest = -1.0;

  // Traverse the array to find the two highest values
  for (int i = 0; i < size; i++) {
    if (arr[i] > firstHighest) {
      secondHighest = firstHighest;
      firstHighest = arr[i];
    } else if (arr[i] > secondHighest && arr[i] < firstHighest) {
      secondHighest = arr[i];
    }
  }
}

#endif
