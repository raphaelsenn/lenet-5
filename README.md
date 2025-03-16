# LeNet-5
Implementation of the deep convolutional neural network called "LeNet-5" described in the paper "Gradient-Based Learning Applied to Document Recognition" in PyTorch.

![image](res/lenet5.png)

## Notes

### Notes about LeNet-5

* **Funfact:** LeNet-5 was many years in commercial use in the NCR Corporation line of check recognition systems for the banking industry. It read millions of checks per month in several banks across the US

* LeNet-5 consists of 7 layers (input -> 1.conv -> 2.sub -> 3.conv -> 4.sub -> 5.conv -> 6.fc -> 7.output)

* Input is a (32 x 32) greyscale images with pixels between the range [-0.1, 1.175] where the background level (white) corresponds to -0.1 and foreground level (black) corresponds to 1.175, this makes the mean input roughly 0 and the variance 1

![image](res/subsampling.png)

### Notes about the paper


### Tradeoff
The gap between the expected error rate on the test set $E_{test}$ and the error rate on the traninig set $E_{train}$ decreases with the number of training samples approximately as:

$$ E_{test} - E_{train} = k(\frac{h}{P})^{\alpha}$$

where $P$ is the number of training samples, $h$ is a measure of "effective capacity", $\alpha \in [0.5, 1]$ and $k$ is a constant.







