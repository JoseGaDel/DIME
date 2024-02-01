#ifndef LOGISTIC_REGRESSION_HEADER_H
#define LOGISTIC_REGRESSION_HEADER_H

#include <algorithm>
#include <climits>

struct findHighestValues {
  void operator()(int8_t arr[], int8_t size, int8_t& firstHighest, int8_t& secondHighest);
};

const float weight1 = 0.03105428;
const float weight2 = -0.00686018;
const float beta = -3.42381517;

int8_t value1;
int8_t value2;

uint8_t LRpredict(int8_t scores[]) {
  findHighestValues()(scores, 10, value1, value2);

  float weighted_sum = beta + weight1 * value1 + weight2 * value2;
  float probability = 1.0 / (1.0 + exp(-weighted_sum));

  if (probability >= 0.5) {
    return 1; // Classify as positive
  } else {
    return 0; // Classify as negative
  }
}

void findHighestValues::operator()(int8_t arr[], int8_t size, int8_t& firstHighest, int8_t& secondHighest) {
  // Initialize the highest and second highest values
  firstHighest = INT8_MIN;
  secondHighest = INT8_MIN;

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
