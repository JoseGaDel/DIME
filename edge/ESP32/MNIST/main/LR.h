#ifndef LOGISTIC_REGRESSION_HEADER_H
#define LOGISTIC_REGRESSION_HEADER_H

#include <algorithm>
#include <cfloat>

struct findHighestValues {
  void operator()(float arr[], uint8_t size, float& firstHighest, float& secondHighest);
};

const float weight1 = 2.10135718;
const float weight2 = -1.8588692;
const float beta = -1.9996633;

float value1;
float value2;

uint8_t LRpredict(float scores[]) {
  findHighestValues()(scores, 10, value1, value2);

  float weighted_sum = beta + weight1 * value1 + weight2 * value2;
  float probability = 1.0 / (1.0 + exp(-weighted_sum));

  if (probability >= 0.5) {
    return 1; // Classify as positive
  } else {
    return 0; // Classify as negative
  }
}

void findHighestValues::operator()(float arr[], uint8_t size, float& firstHighest, float& secondHighest) {
  // Initialize the highest and second highest values
  firstHighest = -FLT_MAX;
  secondHighest = -FLT_MAX;

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
