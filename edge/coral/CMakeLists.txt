cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(measurements)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

set(SOURCES
    cifar.cc
    imagenet.cc
    system.cc
    utils.cc
    image.cc
)

set(FILES )

if (CIFAR10)
    message("CIFAR10 enabled")
    add_definitions(-DCIFAR10=1)
endif()

if (IMAGENET)
    message("IMAGENET enabled")
    add_definitions(-DIMAGENET=1)
endif()

if (SYSTEM)
    message("SYSTEM enabled")
    add_definitions(-DSYSTEM=1)
endif()

if (RESNET8)
    message("RESNET8 enabled")
    add_definitions(-DRESNET8=1)
    list(APPEND FILES
        ${PROJECT_SOURCE_DIR}/models/resnet8_quant_edgetpu.tflite
        ${PROJECT_SOURCE_DIR}/datasets/cifar10/test_batch.bin)
endif()

if (RESNET56)
    message("RESNET56 enabled")
    add_definitions(-DRESNET56=1)
    list(APPEND FILES
        ${PROJECT_SOURCE_DIR}/models/resnet56_quant_edgetpu.tflite
        ${PROJECT_SOURCE_DIR}/datasets/cifar10/test_batch.bin)
endif()

if (ALEXNET)
    message("ALEXNET enabled")
    add_definitions(-DALEXNET=1)
    list(APPEND FILES
        ${PROJECT_SOURCE_DIR}/models/alexnet_quant_edgetpu.tflite
        # NOTE: Here, the batch should be split into two.
        ${PROJECT_SOURCE_DIR}/datasets/cifar10/alexnet_batch1.bin
    )
endif()

if (RESNET18)
    message("RESNET18 enabled")
    add_definitions(-DRESNET18=1)
    list(APPEND FILES
        ${PROJECT_SOURCE_DIR}/models/resnet18_quant_edgetpu.tflite
    )
endif()

if (RESNET50)
    message("RESNET50 enabled")
    add_definitions(-DRESNET50=1)
    list(APPEND FILES
        ${PROJECT_SOURCE_DIR}/models/resnet50_quant_edgetpu.tflite
    )
endif()

add_executable_m7(measurements
    ${SOURCES}
    LINKER_SCRIPT
    ${PROJECT_SOURCE_DIR}/MIMXRT1176xxxxx_cm7_ram.ld
    DATA
    ${FILES}
)

target_link_libraries(measurements
        libs_base-m7_freertos
        libs_base-m7_wifi
)
