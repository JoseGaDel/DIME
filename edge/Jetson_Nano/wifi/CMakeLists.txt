cmake_minimum_required(VERSION 3.10)
project(TCPClient)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

add_executable(TCPClient src/main.cpp)