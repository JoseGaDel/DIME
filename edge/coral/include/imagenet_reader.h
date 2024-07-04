#ifndef IMAGENET_READER_HPP
#define IMAGENET_READER_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "libs/base/filesystem.h"

#define IMAGENET_IMAGE_SIZE 150528

namespace imagenet {
struct imagenet_dataset {
  std::vector<std::vector<uint8_t>> images;

  void resize(std::size_t new_size) {
    if (images.size() > new_size) {
      images.resize(new_size);
    }
  }
};

void read_file(const char *path, imagenet_dataset &dataset, std::size_t limit,
               std::function<std::vector<uint8_t>()> func) {
  if (limit && limit <= dataset.images.size()) {
    return;
  }

  std::vector<uint8_t> data;
  if (!coralmicro::LfsReadFile(path, &data)) {
    while (1) {
      MicroPrintf("no such file: %s", path);
    }
  }

  std::size_t start = dataset.images.size();

  size_t size = 10;
  size_t capacity = limit - dataset.images.size();

  if (capacity > 0 && capacity < size) {
    size = capacity;
  }

  dataset.images.reserve(dataset.images.size() + size);

  for (std::size_t i = 0; i < size; ++i) {
    dataset.images.push_back(func());

    for (std::size_t j = 0; j < IMAGENET_IMAGE_SIZE; ++j) {
      dataset.images[start + i][j] = data[i * IMAGENET_IMAGE_SIZE + j];
    }
  }
}
} // namespace imagenet

#endif // IMAGENET_READER_HPP
