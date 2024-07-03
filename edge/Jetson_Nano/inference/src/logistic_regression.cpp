#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <number_of_classes> <measurement> <iterations>" << std::endl;
        std::cout << "  Where : number_of_classes = {10, 1000}" << std::endl << "        : measurement = {'energy', 'latency'}" << std::endl << "        : iterations = number of iterations to run the loop (defaults to 20,000 iterations)" << std::endl;
        return 1;
    }
    int iterations = 20000;
    if (argc == 4)
    	iterations = std::atoi(argv[3]);


    // Retrieve the integer argument
    int number_of_classes = std::atoi(argv[1]);
    
    // Retrieve the string argument
    std::string measurement = argv[2];
    
    // The following code generates an array of random float values to test the logistic regression
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Define the range

    // Generate the array of random float values
    std::vector<float> classVector(number_of_classes);
    for (int i = 0; i < number_of_classes; ++i) {
        classVector[i] = dis(gen);
    }
    
    // Logistic regression parameters
    float LR_parameters [] = {-4.531622904031753f, 5.82555453f, -3.59687685f};
    
    float logistic = 0.0; // we will accumulate Logistic regression here to prevent the compiler to optimize the loop
                         // away, given that we are operating over the same values.
    float score;

    if (measurement == "latency") {
      auto start_time = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < iterations; it++) {
        float largest = -100.0;
        float secondLargest = -100.0;
        int indexLargest = -1;

        for (int i = 0; i < number_of_classes; ++i) {
            float value = classVector[i];
            if (value > largest) {
                secondLargest = largest;
                largest = value;
                indexLargest = i;
            } else if (value > secondLargest) {
                secondLargest = value;
            }
        }

        // Logistic regression computation
        score = LR_parameters[0] + LR_parameters[1] * largest + LR_parameters[2] * secondLargest; //T score = beta + w1 * largest + w2 * secondLargest;
        logistic += 1 / (1 + exp(-score));
      }
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
      double latency = duration.count() * 1e-6 / iterations;
      
      std::cout << "Average latency for " << iterations << " runs is: " << latency << " milliseconds" << std::endl;
      std::cout << std::endl << "The following is printed to ensure the loop is not optimized away: " << logistic << std::endl;
  } else if (measurement == "energy") {
      // The same, but infinite loop and no time measurement
      while (1) {
        float largest = -100.0;
        float secondLargest = -100.0;
        int indexLargest = -1;

        for (int i = 0; i < number_of_classes; ++i) {
            float value = classVector[i];
            if (value > largest) {
                secondLargest = largest;
                largest = value;
                indexLargest = i;
            } else if (value > secondLargest) {
                secondLargest = value;
            }
        }

        // Logistic regression computation
        score = LR_parameters[0] + LR_parameters[1] * largest + LR_parameters[2] * secondLargest; //T score = beta + w1 * largest + w2 * secondLargest;
        logistic += 1 / (1 + exp(-score));
      }
      
      std::cout << std::endl << "The following is printed to ensure the loop is not optimized away: " << logistic << std::endl; // this won't be printed, but we include it regardless 
                                                                                                                            // so the compiler doesn't perform dead code optimization on the loop
  } else {
    std::cerr << "Error. Choose a valid execution mode (latency / energy)" << std::endl;
    return 1;
  }
  return 0;
}
