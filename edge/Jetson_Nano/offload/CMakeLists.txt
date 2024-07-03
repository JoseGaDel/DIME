cmake_minimum_required(VERSION 3.10)
project(client)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

if (LATENCY)
    add_definitions(-DLATENCY=1)
elseif(ENERGY)
    add_definitions(-DENERGY=1)
endif()

add_executable(client main.cpp)