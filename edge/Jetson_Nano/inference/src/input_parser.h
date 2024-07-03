#pragma once
#include <iostream>
#include "utils.h"

struct CLI_arguments {
    std::string onnxModelPath = "";
    std::string trtModelPath = "";
    std::string precision = "";
    std::string noTF32 = "";
    std::string model = "";
    std::string dataset = "";
    int numIterations = 10000; // Default number of inferences in case it's not provided
    int batch = 1;             // Default batch size in case it's not provided
    int maxBatchSize = 1;      // Default max batch size in case it's not provided. Equal to batch size by default.
};


inline bool parseArguments(int argc, char **argv, CLI_arguments &args) {
    std::string modelPath;

    bool showUsage = false; // Flag to track if usage should be shown. (Parsing anything different from the expected arguments will trigger this flag.)

    // Argument Parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg.substr(0, 8) == "--model=") {
            modelPath = arg.substr(8); // Extract path after "--model="
            args.model = modelPath;
        } else if (arg.substr(0, 13) == "--iterations=") {
            std::string iterationsStr = arg.substr(13);
            try {
                int numIterations = std::stoi(iterationsStr); // Convert to int
                args.numIterations = numIterations; // Convert to int
            } catch (...) {
                std::cerr << "Invalid number of iterations." << std::endl;
            }
        
        } else if (arg.substr(0, 8) == "--batch=") {
            std::string batchStr = arg.substr(8);
            try {
                int batch = std::stoi(batchStr); // Convert to int
                args.batch = batch;
            } catch (...) {
                std::cerr << "Invalid batch size." << std::endl;
            }
        } else if (arg.substr(0, 14) == "--maxBatchSize=") {
            std::string maxBatchSizeStr = arg.substr(14);
            try {
                int maxBatchSize = std::stoi(maxBatchSizeStr); // Convert to int
                args.maxBatchSize = maxBatchSize;
            } catch (...) {
                std::cerr << "Invalid max batch size." << std::endl;
            }
        
        } else if (arg == "--fp16" || arg == "--int8" || arg == "--fp8" || arg == "--best") {
            args.precision = arg.substr(2); // Extract the precision level (removing the "--" from the argument)
        } else if (arg == "--noTF32") {
            args.noTF32 = "noTF32";
        } else if (arg == "-h" || arg == "--help") {
            showUsage = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            showUsage = true;
        }
    }
    // If the user passes a -h or an invalid option, we print the usage cases and terminate the program.
    if (showUsage || modelPath.empty()) { 
        std::cout << "Usage: " << std::endl;
        std::cout << "      --model=<model_path>             : Location of the model to be run. Must have format model_dataset.engine" << std::endl;
        std::cout << "      --iterations=<num_iterations>    : Number of images to test the model with. Default is 10000" << std::endl;
        std::cout << "      --batch=<batch_size>             : Batch size to run the model with. Default is 1" << std::endl;
        std::cout << "      --maxBatchSize=<max_batch_size>  : Maximum batch size to run the model with. Default is equal to batch size" << std::endl;
        std::cout << "      --precision                      : Desired precision level for the engine. From TensorRT 8.6.2 the options are:" << std::endl;
        std::cout << "                        --noTF32  Disable tf32 precision (default is to enable tf32, in addition to fp32)" << std::endl;
        std::cout << "                        --fp16    Enable fp16 precision, in addition to fp32 (default = disabled)" << std::endl;
        std::cout << "                        --int8    Enable int8 precision, in addition to fp32 (default = disabled)" << std::endl;
        std::cout << "                        --fp8     Enable fp8 precision, in addition to fp32 (default = disabled)" << std::endl;
        std::cout << "                        --best    Enable all precisions to achieve the best performance (default = disabled)" << std::endl;
        std::cout << "      -h, --help                       : Show this help message" << std::endl << std::endl;
        std::cout << "Example: " << argv[0] << " --model=../models/resnet18_imagenet.onnx --iterations=10000 --fp16" << std::endl;
        if (modelPath.empty())
            std::cerr << "Please provide a model path using --model=..." << std::endl;
        
        return false; 
    }

    std::string onnx_suffix = ".onnx";
    std::string trt_suffix = ".engine";

    if (modelPath.substr(modelPath.size() - onnx_suffix.size()) == onnx_suffix) {
        if (!Util::doesFileExist(modelPath)) {
            std::cout << "Error: Unable to find model at path '" << modelPath << std::endl;
            return false;
        }
        args.onnxModelPath = modelPath;

    } else if (modelPath.substr(modelPath.size() - trt_suffix.size()) == trt_suffix) {
        if (!Util::doesFileExist(modelPath)) {
            std::cout << "Error: Unable to find engine at path '" << modelPath << std::endl;
            return false;
        }
        args.trtModelPath = modelPath;

    } else {
        std::cerr << "Invalid model path. Must be either an .onnx or .engine file." << std::endl;
        return false;
    }

    // if no precision is provided, check for the presence of a precision in the model path
    // just before the ".engine" extension, like "_fp16.engine"
    if (args.precision.empty()) {
        std::string precision_suffix = "_fp16.engine";
        if (modelPath.substr(modelPath.size() - precision_suffix.size()) == precision_suffix) {
            args.precision = "fp16";
        } else {
            precision_suffix = "_int8.engine";
            if (modelPath.substr(modelPath.size() - precision_suffix.size()) == precision_suffix) {
                args.precision = "int8";
            } else {
                precision_suffix = "_fp8.engine";
                if (modelPath.substr(modelPath.size() - precision_suffix.size()) == precision_suffix) {
                    args.precision = "fp8";
                }
            }
        }
    }

    // Check for 'EE_' and remove it if it exists
    std::string toRemove = "EE_";
    size_t position = modelPath.find(toRemove);
    if (position != std::string::npos) {
        modelPath.erase(position, toRemove.length());
    }

    std::string remove_branch_indicator;
    for (size_t i = 0; i < modelPath.length(); ++i) {
        if (i + 1 < modelPath.length() && modelPath[i] == '_' && (modelPath[i + 1] == '0' || modelPath[i + 1] == '1' || modelPath[i + 1] == '2')) {
            ++i; // Skip both the underscore and the following number
        } else {
            remove_branch_indicator += modelPath[i]; // Append the character to the result
        }
    }
    modelPath = remove_branch_indicator;
    
    // Remove everything from the engine path except the "model_dataset"
    std::string filenameWithExtension = modelPath.substr(modelPath.find_last_of("/") + 1);  // remove everything before the last "/" and the "/" itself
    std::string filenameWithoutExtension = filenameWithExtension.substr(0, filenameWithExtension.find_last_of(".")); // remove the extension
    std::string filenameWithoutPrecision = filenameWithoutExtension.substr(0, filenameWithoutExtension.find_last_of("_")); // remove the precision
    
    // Find the position of the underscore to separate the name of the model and the dataset
    int pos = filenameWithoutPrecision.find('_');
    std::string model, dataset;

    // Check if an underscore was found. If not, print an error message because we need to know the model and dataset.
    if (pos != -1) {
        model = filenameWithoutPrecision.substr(0, pos);
        dataset = filenameWithoutPrecision.substr(pos + 1);
        args.model = model;
        args.dataset = dataset;

        std::cout << "Model: " << model << std::endl;
        std::cout << "Dataset: " << dataset << std::endl;
    } else {
        std::cerr << "Could not determine model and dataset. Check input format." << std::endl;
        return false;
    }

    return true;
}
