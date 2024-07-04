# Coral Dev Board Micro

There are three different setups you can use when compiling the code:
* `CIFAR10`
    * `CIFAR10` goes together with `RESNET8`, `RESNET56` and `ALEXNET`.
* `IMAGENET`
    * `IMAGENET` goes together with `RESNET18` AND `RESNET50`
* `SYSTEM`

For `CIFAR-10`, to run the measurements, the code and the binary containing the
test batch of images is uploaded together to the device. One exception to this
is `AlexNet`, for which the image binary should be split into two (5000 images
each) because of the model's size. The code unpacks the binary and feeds the
images one by one for inference. The result is then supplied to the logistic
regression, which determines whether to offload or not. The output of the
program includes the actual label of an image, inference results, inference
latency, logistic regression result, logistic regression latency.

For `ImageNet-1K`, we cannot store both the model and the binary of images on
device due to resource constraints. Therefore, the directory includes a separate
`client.py` script, that sends one image at a time for the device to infere and
apply logistic regression. The output of the program includes inferred label,
logistic regression result, inference latency and logistic regression latency.

To build the code, see the examples below:

`CIFAR-10` AND `RESNET8`:
```bash
cmake --fresh -DCIFAR10 -DRESNET8=1 -B out -S .
```

`CIFAR-10` AND `RESNET56`:
```bash
cmake --fresh -DCIFAR10 -DRESNET56=1 -B out -S .
```

`CIFAR-10` AND `ALEXNET`:
```bash
cmake --fresh -DCIFAR10 -DALEXNET=1 -B out -S .
```

`IMAGENET` AND `RESNET18`:
```bash
cmake --fresh -DIMAGENET -DRESNET18=1 -B out -S .
```

`IMAGENET` AND `RESNET50`:
```bash
cmake --fresh -DIMAGENET -DRESNET50=1 -B out -S .
```

`SYSTEM`:
```bash
cmake --fresh -DSYSTEM -B out -S .
```

After this, to make, run: `make -C out -j8`. Instead of 8, specify the desired
number of threads.

After compiling the code, it has to be flashed to the device.

To do so, run:

```bash
python3 coralmicro/scripts/flashtool.py --build_dir out --elf_path out/measurements --wifi_ssid <SSID> --wifi_psk <PSK> 
```

Note that for `CIFAR10` measurements, the WiFi is unnecessary, however, for
`IMAGENET`, it is mandatory to supply the network credentials.
