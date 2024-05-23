# Models folder
In this folder we keep all the tested models in their original and quantized versions.

## Resnet 8 (Only for CIFAR-10)
No preprocessing needed. The quantized versions require an int8 input so one has to substract 128 from each element in the original images. In the quantized versions the outputs are also quantized to int8.

A LR model has been implemented for HI decision making. LR is applied to the probability vectors and therefore, the output of the model needs to be unquantized before applying LR.

| Precision | beta                | w1         | w2           |
| --------- | ------------------- | ---------- | ------------ |
| Quant     | -4.63391351         | 5.93658349 | \-3.04197074 |
| Full      | \-4.531622904031753 | 5.82555453 | \-3.59687685 |

## Resnet 56 (Only for CIFAR-10)
This model has been trained with normalized images. Therefore, images need to be preprocessed beforehand. The preprocessing steps are:
- image = image/255.
- image = (image-mean)/std, where mean = [0.4914, 0.4822, 0.4465] and std = [0.2023, 0.1994, 0.2010]
Here the model has been quantized to uint8 for both input and output. Therefore inputs must be quantized accordingly.

This models provide a non-normalized output. So, for the quant version in order to get the confidence values for each class one would have to unquantize the output and add a softmax layer. 

A LR model has been implemented again to check wheter the sample should be offloaded or not based on the two highest confidence values. 
| Precision | beta                | w1         | w2           |
| --------- | ------------------- | ---------- | ------------ |
| Quant     | \-4.695974          | 5.21640018 | \-3.70736749 |
| Full      | \-5.09909203324702  | 5.60475641 | \-4.37203237 |



## Alexnet 
### CIFAR-10
Also trained with normalized images. The required preprocessing in this case is:
- image = (image-mean)/std, where mean = [125.307, 122.95, 113.865] and std = [62.9932, 62.0887, 66.7048].

Here the model has been quantized to uint8 for both input and output. Therefore inputs must be quantized accordingly.

Here, as with Resnet 8 the outputs are normalized. Unquantize them and then aply LR for the offloading decision

| Precision | beta                 | w1         | w2           |
| --------- | -------------------- | ---------- | ------------ |
| Quant     | \-4.33089237         | 5.39107836 | \-1.03445988 |
| Full      | \-3.7969481741850837 | 4.84755215 | \-1.79537036 |

### Imagenet1k

## Resnet18 
The weights used for the logistic regression decision are the following:

| Precision | beta                 | w1         | w2           |
| --------- | -------------------- | ---------- | ------------ |
| Quant     | \-3.02860354         | 4.95192416 | \-1.7800178  |

## Resnet50 

| Precision | beta                 | w1         | w2           |
| --------- | -------------------- | ---------- | ------------ |
| Quant     | \-3.45594814         | 5.11103057 | \-2.07690727 |

## Alexnet

| Precision | beta                 | w1         | w2           |
| --------- | -------------------- | ---------- | ------------ |
| Quant     | \-2.61127983         | 5.00778702 | \-1.23894854 |
