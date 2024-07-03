#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <math.h>
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

    // if no iterations is provided, set to maximum
    if (iterations == 0) {
        iterations = (dataset == "cifar") ? 10000 : 50000;
    }

    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;


    if (dataset == "cifar") {

        // logistic regression parameters
        float LR_parameters [3]; // = { beta, w1, w2 }
        float score, logistic;
        // load the server labels
        std::vector<uint16_t> server_labels = load_labels("../data/cifar_predictions.txt");
        // assign the logistic regression parameters based on the model and precision (full or int8)
        if (model == "resnet8") {
            // if precision == int8: LR_parameters = -4.63391351, 5.93658349, -3.04197074
            // else                : LR_parameters = -4.531622904031753, 5.82555453, -3.59687685 
            LR_parameters[0] = (precision == "int8") ? -4.63391351 : -4.531622904031753;
            LR_parameters[1] = (precision == "int8") ? 5.93658349 : 5.82555453;
            LR_parameters[2] = (precision == "int8") ? -3.04197074 : -3.59687685;

        } else if (model == "resnet56") {
            LR_parameters[0] = (precision == "int8") ? -4.695974 : -5.09909203324702;
            LR_parameters[1] = (precision == "int8") ? 5.21640018 : 5.60475641;
            LR_parameters[2] = (precision == "int8") ? -3.70736749 : -4.37203237;

        } else if (model == "alexnet") {
            LR_parameters[0] = (precision == "int8") ? -4.33089237 : -3.7969481741850837;
            LR_parameters[1] = (precision == "int8") ? 5.39107836 : 4.84755215;
            LR_parameters[2] = (precision == "int8") ? -1.03445988 : -1.79537036;

        } else {
            std::cerr << "Invalid model specified. Supported models are 'resnet8', 'alexnet', and 'resnet56'." << std::endl;
            return 1;
        }

        std::cout << "LR_parameters = " << LR_parameters[0] << ", " << LR_parameters[1] << ", " << LR_parameters[2] << std::endl;

        std::vector<Image> images = LoadCifar10("../data/test_batch.bin");
        // Access image and label
        Image& image = images[0];
        std::vector<uint8_t>& data = image.data;

        const int width = 32;
        const int height = 32;
        const int channels = 3;

        const int inputSize = 32 * 32 * 3;
        float* inputBuffer;
        cudaMallocManaged(&inputBuffer, inputSize * sizeof(float));

        const int outputSize = 10;
        float* outputBuffer;
        cudaMallocManaged(&outputBuffer, outputSize * sizeof(float));

        const auto inputName = engine->getIOTensorName(0);
        const auto outputName = engine->getIOTensorName(1);

        std::cout << "Input name: " << inputName << std::endl << "Output name: " << outputName << std::endl;

        context->setTensorAddress(inputName, inputBuffer);
        context->setTensorAddress(outputName, outputBuffer);

        std::cout << "Primer enqueue" << std::endl;
        // Create a cuda stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        copyImageToInputBufferNCHW(data, inputBuffer, width, height, channels);

        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        std::cout << "Capurando graph .........." << std::endl;
        // Capture a CUDA graph instance
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
        double latency = 0.0;
        int corrects = 0;
        int offloads = 0;
        // Perform inference on each image and record the latency, precision and number of times logistic regression is
        // predicts an offload
        for (int i = 0; i < iterations; ++i) {
            Image& image = images[i];
            uint16_t label = image.label;
            std::vector<uint8_t>& data = image.data;

            if (model == "resnet56") {
                preprocessNormalizeNCHW(data, inputBuffer, resnet56_mean, resnet56_std, height, width, channels);
            } else if (model == "alexnet") {
                preprocessNCHW(data, inputBuffer, alexnet_mean, alexnet_std, height, width, channels);
            } else {
                copyImageToInputBufferNCHW(data, inputBuffer, width, height, channels);
            }
        
            // record the start time and launch graph to perform inference
            auto start_time = std::chrono::high_resolution_clock::now();
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            latency += duration.count();
            
            // logistic regression
            float largest = -100.0;
            float secondLargest = -100.0;
            int indexLargest = -1;

            // if the model is resnet56, apply softmax to output buffer
            if (model == "resnet56") {
                float sum = 0.0;
                for (int i = 0; i < outputSize; ++i) {
                    outputBuffer[i] = exp(outputBuffer[i]);
                    sum += outputBuffer[i];
                }
                for (int i = 0; i < outputSize; ++i) {
                    outputBuffer[i] /= sum;
                }
            }
            // find the largest and second largest values in the output buffer
            // for logistic regression
            for (int i = 0; i < outputSize; ++i) {
                float value = outputBuffer[i];
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
            logistic = 1 / (1 + exp(-score));

            if (logistic < 0.5) {
                offloads++;
                corrects += (server_labels[i] == label);
            } else {
                corrects += (indexLargest == label);
            }
        }
        std::cout << "precision = " << corrects*100.0/iterations << " %" << std::endl;
        std::cout << "offloads = " << offloads << " (" << offloads*100.0/iterations << " %)" << std::endl;
        std::cout << "Latency: " << latency/ (iterations * 1e6) << " ms" << std::endl;


    } else if (dataset == "imagenet") {

        float LR_parameters [3]; // = { beta, w1, w2 }
        float score, logistic;

        if (model == "resnet18") {
            LR_parameters[0] = (precision == "int8") ? -3.02860354 : -4.531622904031753;
            LR_parameters[1] = (precision == "int8") ? 4.95192416 : 5.82555453;
            LR_parameters[2] = (precision == "int8") ? -1.7800178 : -3.59687685;

        } else if (model == "resnet50") {
            LR_parameters[0] = (precision == "int8") ? -3.45594814 : -5.09909203324702;
            LR_parameters[1] = (precision == "int8") ? 5.11103057 : 5.60475641;
            LR_parameters[2] = (precision == "int8") ? -2.07690727 : -4.37203237;

        } else if (model == "alexnet") {
            LR_parameters[0] = (precision == "int8") ? -2.61127983 : -3.7969481741850837;
            LR_parameters[1] = (precision == "int8") ? 5.00778702 : 4.84755215;
            LR_parameters[2] = (precision == "int8") ? -1.23894854 : -1.79537036;

        } else {
            std::cerr << "Invalid model specified. Supported models are 'resnet18', 'alexnet', and 'resnet50'." << std::endl;
        }

        const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        const std::vector<float> std = {0.229f, 0.224f, 0.225f};


        const int width = 224;
        const int height = 224;
        const int channels = 3;

        const int inputSize = 224 * 224 * 3;
        const int outputSize = 1000;

        // load server labels
        std::vector<uint16_t> server_labels = load_labels("../data/imagenet_predictions.txt");

        std::vector<uint8_t> imageData(width * height * channels);

        // load the ground truth labels
        std::vector<uint16_t> labels = load_labels("../data/imagenet_labels.txt");
        

        float* inputBuffer; 
        cudaMallocManaged(&inputBuffer, inputSize * sizeof(float));

        float* outputBuffer;
        cudaMallocManaged(&outputBuffer, outputSize * sizeof(float), cudaMemAttachHost);

        std::ostringstream filename_binary;
        filename_binary << "../data/bin/ILSVRC2012_val_" << std::setw(8) << std::setfill('0') << 1 << ".bin";
        
        std::ifstream file(filename_binary.str(), std::ios::binary);
        if (!file) {
            std::cerr << "Unable to open the binary file." << std::endl;
            return 1;
        }
        file.read(reinterpret_cast<char*>(imageData.data()), inputSize * sizeof(uint8_t));
        file.close();

        preprocessImage(imageData, inputBuffer, width, height, channels);
        cudaStreamAttachMemAsync(NULL, inputBuffer, 0, cudaMemAttachGlobal);

        const auto inputName = engine->getIOTensorName(0);
        const auto outputName = engine->getIOTensorName(1);

        context->setTensorAddress(inputName, inputBuffer);
        context->setTensorAddress(outputName, outputBuffer);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        cudaStreamAttachMemAsync(NULL, outputBuffer, 0, cudaMemAttachHost);
        cudaStreamSynchronize(NULL);

        std::cout << "Capurando graph .........." << std::endl;

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
        double latency = 0.0;
        int corrects = 0;
        int offloads = 0;

        for (int i = 0; i < iterations; ++i) {
            
            std::ostringstream filename;
            filename << "../data/bin/ILSVRC2012_val_" << std::setw(8) << std::setfill('0') << i+1 << ".bin";

            load_bin(filename.str(), imageData);

            uint16_t label = labels[i];
            
            preprocessImage(imageData, inputBuffer, width, height, channels);
            auto start_time = std::chrono::high_resolution_clock::now();
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            latency += duration.count();

            // logistic regression
            float largest = -100.0;
            float secondLargest = -100.0;
            int indexLargest = -1;

            for (int i = 0; i < outputSize; ++i) {
                float value = outputBuffer[i];
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
            logistic = 1 / (1 + exp(-score));

            if (logistic < 0.5) {
                offloads++;
                corrects += (server_labels[i] == label);
            } else {
                corrects += (indexLargest == label);
            }
        }
        std::cout << "precision = " << corrects*100.0/iterations << " %" << std::endl;
        std::cout << "offloads = " << offloads << " (" << offloads*100.0/iterations << " %)" << std::endl;
        std::cout << "Latency: " << latency/ (iterations * 1e6) << " ms" << std::endl;
    } else {
        std::cerr << "Invalid dataset specified. Supported datasets are 'cifar' and 'imagenet'." << std::endl;
        return 1;
    }
    return 0;
    
};
