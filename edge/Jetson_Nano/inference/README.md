This directory contains the programs designed to measure accuracy, latency and energy consumption of a model running on the Jetson Orin Nano. To perform inference, we are using TensorRT, which is a framework developed by NVIDIA for high-performance deep learning inference, built on top of CUDA. It optimizes the models using layer and tensor fusion and kernel tuning to maximize performance on the specific hardware of the target device. Jetson Orin has an Ampere GPU and therefore includes Tensor cores, which are programmable fused matrix-multiply-and-accumulate units that execute concurrently alongside the CUDA cores and support half-precision and integer instructions. This hardware works great with operations common in CNNs and benefit greatly from optimization techniques like INT8 quantization and fine-grained structured sparsity.

TensorRT allows for asynchronous execution of multiple inference streams and profiles of the same model optimized for different batch sizes to implement opportunistic batching. Still, for our case study we are doing single inferences with busy waiting between subsequent calls. This device has a couple peculiarities that influenced implementation decisions, which are discussed in the following paragraphs. You may jump to the [Usage](#usage) section to see how to run the programs.

## Unified Memory

The Jetson Orin Nano has an integrated GPU, which means we have unified memory, i.e, both CPU and GPU have access to the same physical memory address space. A descrete GPU would require to copy data from host to device memory before processing, and then copy the results back to host memory. This is not the case for the Jetson Orin Nano, and consequently, a zero-copy approach has been implemented to avoid superfluous memory transfers between host and device. For this, when allocating buffers, we use Managed Memory, which is a CUDA feature that allows the programmer to allocate memory that is accessible from both the CPU and the GPU. This is done by calling `cudaMallocManaged` instead of `cudaMalloc` and then using the pointer returned by this function to access the memory. This pointer refers to memory that is visible to both the CPU and the GPU, so we can pass that pointer to write directly from the CPU:

```c
float* inputBuffer;
cudaMallocManaged(&inputBuffer, inputSize * sizeof(float));
```

This memory is attached to the GPU when the GPU accesses it and detached when the CPU accesses it. This means that the first access to the memory from the GPU will cause a page fault, which will trigger the data to be copied to the GPU. Subsequent accesses from the GPU will not cause a page fault, as the data is already in the GPU's memory. This is done transparently by the CUDA runtime, so the programmer does not need to worry about it. In our case, we are performing preprocessing on the CPU, but we could optimize this further by moving the preprocessing to the GPU. Then, the input buffer would only be accessed by the GPU, we could allocate them on device with `cudaMalloc` and not worry about I/O coherency. For more information, check the [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management) programming guide, specifically section 4. The output buffer will be accessed by the CPU, so we need to allocate it with `cudaMallocManaged` and we attach it to the CPU as so:

```c
float* outputBuffer;
cudaMallocManaged(&outputBuffer, outputSize * sizeof(float), cudaMemAttachHost);
```

Applications that use unified memory have to conduct extra coherency checks and cache maintenance procedures during kernel startup, synchronization, and prefetching hint calls. Because these tasks are completed concurrently with other GPU tasks, the application may experience irregular latencies. Data prefetching hints can be used to improve performance in this paradigm. These prefetching tips can help the driver optimize the coherence operations. This is why our implementation includes lines like:

```c
cudaStreamAttachMemAsync(NULL, inputBuffer, 0, cudaMemAttachGlobal); // Here we are prefetching the data to the GPU

cudaStreamAttachMemAsync(NULL, outputBuffer, 0, cudaMemAttachHost); // Here we are prefetching the data to the CPU
cudaStreamSynchronize(NULL);
```

## CUDA Graphs

If you just invoke the TensorRT routine to perform inference, wether is `enqueueV3()` or `execute()`, you will notice that the GPU compute time is small compared with host walltime. In fact, `trtexec` will warn you about this and suggest you set the flag `--useCudaGraph` to improve performance. This is because the duration of the enqueue calls of the TensorRT execution context is a bottleneck for inference performance. The reason is that the network contains many layers with short kernel execution time, causing the kernel launch time to dominate over the inference latency. This can be mitigated using CUDA Graphs, which provide a way to record the GPU kernels invoked by a program into a graph data structure, which can be later launched from the application in a single operation reducing the overhead. This also allows overlapping kernel launches, execution and memory copies. In this project it has been implemented by capturing the execution of one inference call and then launching the graph in subsequent calls instead of the original enqueue call. The graph is captured with:

```c
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// launch one inference instance to record the graph
context->enqueueV3(stream);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
```

And then, each time we want to perform inference, we launch the graph with:

```c
cudaGraphLaunch(instance, stream);
cudaStreamSynchronize(stream);
```

Note that we enforce synchronization after launching the graph to ensure that the graph is executed before the next inference call. This is because the graph is launched asynchronously, so the function returns control to the main program and it will continue running unless we impose synchronization. We have opted for single inferences, but CUDA Graphs can be used to capture more complex workflows, including multiple streams and dependencies between them.

## Power Management

The Jetson Orin Nano provides power and performance management features giving the user the ability to tweak the device to their needs. For example, the `cpufreq` subsystem allows for dynamic voltage and frequency scaling (DVFS), where clockrate and voltage is adjusted in real time to instant load. Jetson Orin Nano is able to operate in two power modes: one to maximize performance with a power budget of 15 W, where all hardware is available and clocks can reach their maximum values, and another restrained mode of 7 W where memory, CPU, GPU, etc is capped to confine the power draw to the target. If we want to configure the device for maximum performance, we need to run the following command on the terminal:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

If you want to set the device to the 7 W mode, just replace the first part with `sudo nvpmodel -m 0`. For our measurements, we have used both power modes to explore how they affect the behavior of the device in this context. We observed that allowing the DVFS governor to dynamically adjust clock rates introduced a significant hit on latency and variability for a relatively small save in energy consumption. A decision was taken to lock the clocks to the maximum of each power mode, given that in our tests the device was under constant load and performance per watt was noticeably better and results were more stable. A user, however, may decide to allow dynamic clock scale if the computations are spread such that the device will be idle most of the time. If that is the case, remove the `sudo jetson_clocks` command, which is responsible to lock the clocks to the maximum.

## Usage

The program `src/main.cpp` performs inference on a TRT engine located in `models/`, measures accuracy of the selected model and average latency using high-precision timers using the `std::chrono` library. Given that the inference runtime was so short, it was observed that the overhead of measuring execution time in each iteration was significant, so latency can be more accurately measured by running `src/measurements.cpp`. This one measures outside a loop that executes multiple times and divides the total time by the number of iterations, producing a more accurate result. That same program can run the loop indefinitely to measure energy consumption. To choose which version gets compiled, we need to set the `LATENCY` and `ENERGY` flags in the CMake configuration. For example, to measure latency and not energy consumption, run the following command:

```bash
mkdir build
cd build
cmake -DLATENCY=1 -DENERGY=0 ..
```

Then, compile the program with `make`. You will get the following executables:

* `inference` to measure accuracy and latency
* `measurements` to measure latency if you compiled with `-DLATENCY=1`, or energy consumption if you compiled with `-DENERGY=1`
* `simulation` to run a simulation of the HI system. In this repo, we have implemented a simplified version if you are only concerned with the numbers shown in the paper. Therefore, the program instead of actually performing the offload, reads from a file that contains the class that the server model is going to produce, and use that instead. If you want to run the actual system, you may use the code in [the client program in the offloading section](../offload).
* `logistic_regression` to test latency and energy consumption of logistic regression on the output of the model. This simply makes a random vector of the desired size and measures any of those two metric in a similar manner to the other programs.

To run the first three, you need to pass the path to the TRT engine (explained next). The name must have format `name_dataset_precision.engine`. We can also indicate the number of iterations we want to run the program with the flag `--iterations`. If not specified, the program will default to 10000 iterations. For example, we can run the program with the following command:

```bash
./inference --model=../models/resnet56_imagenet_int8.engine
```

Or for example, to run the measurements program with 50000 iterations, run

```bash
./measurements --model=../models/regnet_imagenet_int8.engine --iterations=50000
```


To run the last, run with `./LogisticRegression <number_of_classes> <measurement>`. So, for example, to measure latency for CIFAR-10, run 

```bash
./LogisticRegression 10 latency
```
Or if you want to measure energy consumption for ImageNet1k, run

```bash
./LogisticRegression 1000 energy
```


### Building a model engine

On the models directory you can find some ONNX models. To build an engine, you can either use `trtexec`, which comes with the TensorRT framework, or use the program [build_engine.py in the int8/ directory](int8/build_engine.py). The latter is required to build engines with INT8 quantization, as `trtexec` does not support INT8 calibration. To build an engine with `trtexec`, go to the models directory and run the following command:

```bash
trtexec --builderOptimizationLevel=99 --skipInference --onnx=model.onnx --saveEngine=model_dataset_precision.engine --precision
```

The flag `--precision` should actually only be `--fp16`. If you want to compile to full FP32, you can remove the flag. If you want to compile to INT8, you need to use the script. If you don't care about precision and don't want to perform entropy calibration, you can use `trtexec` `--int8` flag (or `--best` to allow fallback to FP16 for those layers that do not support INT8). If you want to use the script, go to the int8 directory and run the following command. For example, to build ResNet-8 with int8 precission, we need to indicate the path to the onnx model, the path to the engine we want to save, the precision we want to use, the path to the calibration cache, the batch size for calibration, the dataset we are using, the number of images to use for calibration, and the number of images to use for validation. For example, to build an engine for ResNet-8:

```bash
python3 build_engine.py --onnx ../models/resnet8_cifar.onnx --engine ../models/resnet8_cifar_int8.engine --precision int8 --calib_cache ./calibration.cache --calib_batch_size 2500 --dataset cifar --calib_num_images 10000
```

However, the program can extract most of the information from the name of the onnx model, so you can just run:

```bash
python3 build_engine.py --onnx ../models/resnet8_cifar.onnx --engine ../models/resnet8_cifar_int8.engine --precision int8 --calib_batch_size 2500 --calib_num_images 10000
```
