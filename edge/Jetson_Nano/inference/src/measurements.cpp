#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include "utils.h"
#include "input_parser.h"

// Preprocessing parameters
const std::vector<float> resnet56_mean = {0.4914f, 0.4822f, 0.4465f};
const std::vector<float> resnet56_std = {0.2023f, 0.1994f, 0.2010f};
const std::vector<float> alexnet_mean = {125.307f, 122.95f, 113.865f};
const std::vector<float> alexnet_std = {62.9932f, 62.0887f, 66.7048f};




void saveEngineToDisk(const std::string& onnxFileName, const std::string& engineName)
{
    std::cout << "[INFO] " << "Creating builder and network..." << std::endl;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    std::cout << "[INFO] " << "Loading ONNX file in the disk..." << std::endl;
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    std::cout << "[INFO] " << "Building cuda engine..." << std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 30); // 60 Mb
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    std::cout << "[INFO] " << "Serializing engine..." << std::endl;
    nvinfer1::IHostMemory* serializedModel = engine->serialize();

    std::cout << "[INFO] " << "Saving serialized model in the disk..." << std::endl;
    std::ofstream ofs(engineName, std::ios::binary | std::ios::out);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();

    serializedModel->destroy();
};



nvinfer1::ICudaEngine* loadEngineFromDisk(const std::string& engineName)
{
    std::cout << "Loading " << engineName << " from the disk." << std::endl;
    std::ifstream ifs(engineName, std::ios::binary);

    if (!ifs.good())
    {
        std::cout << "Please check the path of your engine file." << std::endl;
        exit(-1);
    }

    size_t size = 0;

    ifs.seekg(0, ifs.end);
    size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char* trtModelStream = new char[size];

    ifs.read(trtModelStream, size);
    ifs.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);

    delete[] trtModelStream;
    return engine;
};



int main(int argc, char* argv[]) {
    CLI_arguments arguments;

    if (!parseArguments(argc, argv, arguments)) {
        return 1;
    }

    nvinfer1::ICudaEngine* engine = loadEngineFromDisk(arguments.trtModelPath);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    std::string model = arguments.model;
    std::string dataset = arguments.dataset;
    std::string precision = arguments.precision;
    int iterations = arguments.numIterations;


    if (dataset == "cifar") {

        std::vector<Image> images = LoadCifar10("../data/test_batch.bin");
        // Access image and label
        Image& image = images[0];
        std::vector<uint8_t>& data = image.data;

        images.clear();

        const int width = 32;
        const int height = 32;
        const int channels = 3;

        // CIFAR-10 input shape is 1x32x32x3
        const int inputSize = 32 * 32 * 3;
        float* inputBuffer;
        cudaMallocManaged(&inputBuffer, inputSize * sizeof(float)); // buffer attached to the GPU

        // CIFAR-10 output shape is 1x10
        const int outputSize = 10;
        float* outputBuffer;
        cudaMallocManaged(&outputBuffer, outputSize * sizeof(float), cudaMemAttachHost); // buffer attached to the CPU

        const auto inputName = engine->getIOTensorName(0);
        const auto outputName = engine->getIOTensorName(1);

        context->setTensorAddress(inputName, inputBuffer);
        context->setTensorAddress(outputName, outputBuffer);

        // Create a cuda stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        if (model == "resnet56") {
            preprocessNormalizeNCHW(data, inputBuffer, resnet56_mean, resnet56_std, height, width, channels);
        } else if (model == "alexnet") {
            preprocessNCHW(data, inputBuffer, alexnet_mean, alexnet_std, height, width, channels);
        } else {
            copyImageToInputBufferNCHW(data, inputBuffer, width, height, channels);
        }

        cudaStreamAttachMemAsync(NULL, inputBuffer, 0, cudaMemAttachGlobal); // Prefetch the inputBuffer to the GPU

        context->enqueueV3(stream); // Call enqueueV3() once after an input shape change to update internal state.
        cudaStreamSynchronize(stream);

        cudaStreamAttachMemAsync(NULL, outputBuffer, 0, cudaMemAttachHost); // Prefetch the outputBuffer to the CPU
        cudaStreamSynchronize(NULL);

        std::cout << "Capturing CUDA graph .........." << std::endl;
        // Capture a CUDA graph instance to optimize kernel launches
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // launch one inference instance to record the graph
        context->enqueueV3(stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

        // perform a warmup to avoid cold start

        std::cout << std::endl << "_____________________" << std::endl << "|  Performing warmup  |" << std::endl << "_____________________" << std::endl;

        for (int i = 0; i < 100; ++i) {
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        }

        std::cout << std::endl << "________________________" << std::endl << "|  Performing inference  |" << std::endl << "________________________" << std::endl;
#ifdef LATENCY
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
                cudaGraphLaunch(instance, stream);
                cudaStreamSynchronize(stream);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double latency = duration.count();	
	    std::cout << "Latency: " << latency/ (iterations * 1e6) << " ms" << std::endl;
#endif

#ifdef ENERGY
        // To run inferences, launch the graph instead of calling enqueueV3().
        while (true) {
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        }
#endif

    } else if (dataset == "imagenet") {
        // imagenet are 224x224x3 images, so if pixels are 8-bit unsigned integers, each image is 150528 bytes. Given
        // that there are 50,000 images in the validation set, the total size of the dataset is 7.5264 GB. The Jetson Orin
        // has 8 GB of main memory, so we have to load the images in batches. We will use a batch size of 10000 images, so
        // unless the user specifies a number of iterations lower, in which case we will load the number of images specified

        const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        const std::vector<float> std = {0.229f, 0.224f, 0.225f};


        const int width = 224;
        const int height = 224;
        const int channels = 3;

        const int inputSize = 224 * 224 * 3;
        const int outputSize = 1000;

        std::vector<uint8_t> imageData(width * height * channels);

        float* inputBuffer; 
        cudaMallocManaged(&inputBuffer, inputSize * sizeof(float)); // buffer attached to the GPU

        float* outputBuffer;
        cudaMallocManaged(&outputBuffer, outputSize * sizeof(float), cudaMemAttachHost); // buffer attached to the CPU

        std::ostringstream filename_binary;
        filename_binary << "../data/bin/ILSVRC2012_val_" << std::setw(8) << std::setfill('0') << 1 << ".bin";
        
        // load binary in data and png in imageData
        std::ifstream file(filename_binary.str(), std::ios::binary);
        if (!file) {
            std::cerr << "Unable to open the binary file." << std::endl;
            return 1;
        }
        file.read(reinterpret_cast<char*>(imageData.data()), inputSize * sizeof(uint8_t));
        file.close();

        preprocessImage(imageData, inputBuffer, width, height, channels); // fill the inputBuffer with the preprocessed image

        cudaStreamAttachMemAsync(NULL, inputBuffer, 0, cudaMemAttachGlobal); // Prefetch the inputBuffer to the GPU

        const auto inputName = engine->getIOTensorName(0);
        const auto outputName = engine->getIOTensorName(1);

        context->setTensorAddress(inputName, inputBuffer);
        context->setTensorAddress(outputName, outputBuffer);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        cudaStreamAttachMemAsync(NULL, outputBuffer, 0, cudaMemAttachHost); // Prefetch the outputBuffer to the CPU
        cudaStreamSynchronize(NULL);

        std::cout << "Capuring graph .........." << std::endl;

        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        context->enqueueV3(stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

        std::cout << std::endl << "_____________________" << std::endl << "|  Performing warmup  |" << std::endl << "_____________________" << std::endl;

        for (int i = 0; i < 100; ++i) {
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        }

        std::cout << std::endl << "________________________" << std::endl << "|  Performing inference  |" << std::endl << "________________________" << std::endl;
#ifdef LATENCY
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
                cudaGraphLaunch(instance, stream);
                cudaStreamSynchronize(stream);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double latency = duration.count();
        std::cout << "Latency: " << latency/ (iterations * 1e6) << " ms" << std::endl;
#endif

#ifdef ENERGY
        while (true) {
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        }
#endif
    } else {
        std::cerr << "Invalid dataset specified. Supported datasets are 'cifar' and 'imagenet'." << std::endl;
        return 1;
    }
    return 0;
    
};
