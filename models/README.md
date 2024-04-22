# Models folder
In this folder we keep all the tested models in their original and quantized versions.

## Resnet 8 and variations (Only for CIFAR-10)
No preprocessing needed. The quantized versions require an int8 input so one has to substract 128 from each element in the original images. In the quantized versions the outputs are also quantized to int8.

A LR model has been implemented for HI decision making. The regression has been trained directly with the quantized outputs which is equivalent due to the linearity of the expression given that only a binary answer is needed. The parameters for this model are: beta = 3.42381517, w1 = 0.03105428 and w2 = -0.00686018

| Precision | beta                | w1         | w2           |
| --------- | ------------------- | ---------- | ------------ |
| Quant     | 3.42381517          | 0.03105428 | \-0.00686018 |
| Full      | \-4.531622904031753 | 5.82555453 | \-3.59687685 |

## Resnet 56 and variations
### CIFAR-10
This model has been trained with normalized images. Therefore, images need to be preprocessed beforehand. The preprocessing steps are:
- image = image/255.
- image = (image-mean)/std, where mean = [0.4914, 0.4822, 0.4465] and std = [0.2023, 0.1994, 0.2010]
Here the model has been quantized to uint8 for both input and output. Therefore inputs must be quantized accordingly. The parameters can be found using: "interpreter.get_input_details()" and they are: scale = 0.02032469 and zero = 120

This models provide a non-normalized output. So, for the quant version in order to get the confidence values for each class one would have to dequantize the output (scale = 0.18410155 and zero = 74) and add a softmax layer. 

A LR model has been implemented to check wheter the sample should be offloaded or not based on the two highest confidence values. The parameters are: beta = -4.50152442, w1 = 5.02825697 and w2 = -3.943078

| Precision | beta                | w1         | w2           |
| --------- | ------------------- | ---------- | ------------ |
| Quant     | \-4.50152442        | 5.02825697 | \-3.943078   |
| Full      | \-5.09909203324702  | 5.60475641 | \-4.37203237 |

### Imagenet1k

## Alexnet and variations
### CIFAR-10
Also trained with normalized images. The required preprocessing is:
- image = (image-mean)/std, where mean = [125.307, 122.95, 113.865] and std = [62.9932, 62.0887, 66.7048].

Here the model has been quantized to uint8 for both input and output. Therefore inputs must be quantized accordingly. The parameters can be found using: "interpreter.get_input_details()" and they are: scale = 0.016141219064593315 and zero = 123

Here, as with Resnet 8 the outputs are already normalized. Therefore, the LR model is trained directly with the quant outputs. The trained parameters are: beta = -4.58669089, w1 = 0.02206441 and w2 = -0.00222133

| Precision | beta                 | w1         | w2           |
| --------- | -------------------- | ---------- | ------------ |
| Quant     | \-4.58669089         | 0.02206441 | \-0.00222133 |
| Full      | \-3.7969481741850837 | 4.84755215 | \-1.79537036 |

### Imagenet1k

## Resnet18 (Only for Imagenet1k)


## Resnet50 (Only for Imagenet1k)
