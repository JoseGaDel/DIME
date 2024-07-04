# DIME

This repository contains the implementation of the measurements presented in the paper:

> **Exploring the Boundaries of On-Device Inference: When Tiny Falls Short, Go Hierarchical**<br>

Here we release the code and datasets used for analyzing the Hierarchical Inference (HI) system, a method to improve on-device ML inference by offloading challenging samples to an edge server or cloud for remote inference. For embedded ML models, HI seeks to raise accuracy, minimize latency, and cut energy usage. We evaluate these performance criteria over several devices and models. It also includes the Early Exit with HI (EE-HI) system, which maximizes energy economy and latency still more.

On the `edge/` directory, you will find the code for the edge devices, including the ESP32, Arduino Nano 33, Raspberry Pi, Coral Micro, and Jetson Orin Nano. The `server/` directory contains the code for the server-side, which is used to offload the samples from the edge. The `models/` directory contains the models used in common between multiple devices. Jetson Orin contains some models, as it uses a different framework.