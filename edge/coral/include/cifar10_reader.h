#ifndef CIFAR10_READER_HPP
#define CIFAR10_READER_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "libs/base/filesystem.h"

#define IMAGE_SIZE 3073

namespace cifar {
struct CIFAR10_dataset {
  std::vector<std::vector<uint8_t>> images;
  std::vector<uint8_t> labels;

  void resize(std::size_t new_size) {
    if (images.size() > new_size) {
      images.resize(new_size);
      labels.resize(new_size);
    }
  }
};

void read_file(const char *path, CIFAR10_dataset &dataset, std::size_t limit,
               std::function<std::vector<uint8_t>()> func) {
  if (limit && limit <= dataset.images.size()) {
    return;
  }

  std::vector<uint8_t> data;
  if (!coralmicro::LfsReadFile(path, &data)) {
    while (1) {
      printf("no such file: %s\r\n", path);
    }
  }

  std::size_t start = dataset.images.size();

  size_t size = 10000;
  size_t capacity = limit - dataset.images.size();

  if (capacity > 0 && capacity < size) {
    size = capacity;
  }

  // Prepare the size for the new
  dataset.images.reserve(dataset.images.size() + size);
  dataset.labels.resize(dataset.labels.size() + size);

  for (std::size_t i = 0; i < size; ++i) {
    dataset.labels[start + i] = data[i * IMAGE_SIZE];

    dataset.images.push_back(func());

    for (std::size_t j = 1; j < IMAGE_SIZE; ++j) {
      dataset.images[start + i][j - 1] = data[i * IMAGE_SIZE + j];
    }
  }
}
} // namespace cifar

#endif
