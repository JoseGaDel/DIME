# Coral Dev Board Micro

For `CIFAR-10` measurements, make sure to change the `SOURCES` in
`CMakeLists.txt` so that it includes `cifar.cc`. For `ImageNet-1K`,
`imagenet.cc` and for offloading measurements, `system.cc` should be included.

Note that for all three cases, the setups are different.

For `CIFAR-10`, to run the measurements, the code and the binary containing the
test batch of images is uploaded together to the device. One exception to this
is `AlexNet`, for which the image binary should be split into two (5000 images
each). The code unpacks the binary and feeds the images one by one for
inference. The result is then supplied to the logistic regression, which
determines whether to offload or not. The output of the program includes the
actual label of an image, inference results, inference latency, logistic
regression result, logistic regression latency.

For `ImageNet-1K`, we cannot store both the model and the binary of images on
device due to resource constraints. Therefore, the directory includes a separate
`client.py` script, that sends one image at a time for the device to infere and
apply logistic regression. The output of the program includes inferred label,
logistic regression result, inference latency and logistic regression latency.

To build the code, see the below example. Available model options for building
the code are: `RESNET8`, `RESNET56`, `ALEXNET`, `RESNET18` and `RESNET50`.


```bash
cmake --fresh -DRESNET8=1 -B out -S .
make -C out
```

After compiling the code, it has to be flashed to the device.
