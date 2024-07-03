#
# This script builds a TensorRT engine from an ONNX model. It supports building engines in FP32, FP16 and INT8 precision.
# For INT8 precision, it requires a directory of images to use for calibration. The program is built from the TensorRT samples
# which can be found at:
# https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/
#

import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

from image_batcher import ImageBatcher, CIFAR10BinaryBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(
            np.dtype(self.image_batcher.dtype).itemsize
            * np.prod(self.image_batcher.shape)
        )
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info(
                "Calibrating image {} / {}".format(
                    self.image_batcher.image_index, self.image_batcher.num_images
                )
            )
            common.memcpy_host_to_device(
                self.batch_allocation, np.ascontiguousarray(batch)
            )
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 8 * (2**30)
        )  # 8 GB

        self.profile = self.builder.create_optimization_profile();
        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """

        self.network = self.builder.create_network(self.EXPLICIT_BATCH)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        #assert self.batch_size > 0

    def create_engine(
        self,
        engine_path,
        precision,
        calib_input=None,
        calib_cache=None,
        calib_num_images=35000,
        calib_batch_size=512,
        calib_preprocessor=None,
        engine_batch_size=1,
        dataset="imagenet",
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        log.info("Reading timing cache from file: {:}".format(args.timing_cache))
        common.setup_timing_cache(self.config, args.timing_cache)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                if 'EE_' not in engine_path:
                    self.config.set_flag(trt.BuilderFlag.FP16) # for fallback to FP16 instead of FP32 in those layers that cannot be quantized
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    if dataset == "imagenet":
                        self.config.int8_calibrator.set_image_batcher(
                            ImageBatcher(
                                calib_input,
                                calib_shape,
                                calib_dtype,
                                max_num_images=calib_num_images,
                                exact_batches=True,
                                preprocessor=calib_preprocessor,
                            )
                        )
                    elif dataset == "cifar":
                        self.config.int8_calibrator.set_image_batcher(
                            CIFAR10BinaryBatcher(
                                calib_input,
                                calib_shape,
                                calib_dtype,
                                batch_size=calib_batch_size,
                                preprocessor=calib_preprocessor,
                            )
                        )
                    else:
                        log.error("Unknown dataset for INT8 calibration. Supported datasets: 'imagenet', 'cifar'")
                        sys.exit(1)

        network_inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        for _input in network_inputs:
            # set the shape of the input tensor with batch size = engine_batch_size
            _input.shape = (engine_batch_size,) + tuple(_input.shape[1:])
            self.profile.set_shape(_input.name, _input.shape, _input.shape, _input.shape)
        
        self.config.add_optimization_profile(self.profile)
        self.config.builder_optimization_level = 99 # Allow TensorRT to spend more time optimizing the engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        log.info("Serializing timing cache to file: {:}".format(args.timing_cache))
        common.save_timing_cache(self.config, args.timing_cache)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size,
        args.calib_preprocessor,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )
    parser.add_argument(
        "--calib_input", help="The directory holding images to use for calibration"
    )
    parser.add_argument(
        "--calib_cache",
        help="The file path for INT8 calibration cache to use, default: ./model_name.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=25000,
        type=int,
        help="The maximum number of images to use for calibration, default: 25000",
    )
    parser.add_argument(
        "--calib_batch_size",
        default=512,
        type=int,
        help="The batch size for the calibration process, default: 512",
    )
    parser.add_argument(
        "--engine_batch_size",
        default=1,
        type=int,
        help="The batch size for the compiled engine, default: 1",
    )
    parser.add_argument(
        "--calib_preprocessor",
        choices=["resnet8", "resnet56", "alexnet"],
        help="Set the calibration image preprocessor to use, either 'resnet8', 'resnet56' or 'alexnet' for CIFAR-10. For ImageNet, it will use the same for every model",
    )
    parser.add_argument(
        "--timing_cache",
        default="./timing.cache",
        help="The file path for timing cache, default: ./timing.cache",
    )
    parser.add_argument(
        "--dataset",
        #default="imagenet",
        choices=["imagenet", "cifar"],
        help="The dataset to use for INT8 calibration, either 'imagenet' or 'cifar'. If not provided, will take from argument --dataset or from model name if present",
    )

    args = parser.parse_args()
    if not args.onnx:
        parser.print_help()
        log.error("These arguments are required: --onnx")
        sys.exit(1)

    # check if the model name has _cifar_ or _imagenet_ in it and override the value of dataset argument
    if "_cifar" in args.onnx or "_cifar" in args.engine:
        args.dataset = "cifar"
    elif "_imagenet" in args.onnx or "_imagenet_" in args.engine:
        args.dataset = "imagenet"
    if not args.engine:
        args.engine = os.path.splitext(args.onnx)[0] + "_" + args.precision + ".engine"
    print(f'\ncalib_input: {args.calib_input}\n')
    # if no calib input is specified, check dataset and assign default for each
    if not args.calib_input:
        if args.dataset == "cifar":
            args.calib_input = "../data/test_batch.bin"
        elif args.dataset == "imagenet":
            args.calib_input = "../data/imagenet/"
    # if no calib_preprocessor is specified, assign the name of the model which will be the characters between the last / and the first _
    if not args.calib_preprocessor:
        args.calib_preprocessor = args.onnx.split("/")[-1].split("_")[0]

    if not args.calib_cache:
        args.calib_cache = args.onnx.split("/")[-1].split("_")[0] + "_" + args.dataset + ".cache"

    # print all args:
    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")
    main(args)
