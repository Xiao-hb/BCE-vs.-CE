# ![Logo image](https://github.com/user-attachments/assets/26a93348-5bed-4107-9113-0dfe6ee707e8) \[ICML 2025\] BCE vs. CE in Deep Feature Learning

This is the **main code** for the paper "[BCE vs. CE in Deep Feature Learning]()".

Forty-Second International Conference on Machine Learning \(ICML\), 2025.

🎬 None | 💻 None | 🔥 [Poster]().

# Overview
* We provide the first theoretical proof that BCE can also lead to the NC, i.e., maximizing the compactness and distinctiveness.
* We find that BCE performs better than CE in enhancement of intra-class compactness and inter-class distinctiveness across sample features, and, BCE can explicitly enhance the feature properties, while CE only implicitly enhance them.
* We point out that when training models with BCE, the classifier biases play substantial role in enhancing the feature properties, while in the training with CE, it almost does not work.
* We conduct extensive experiments, and find that, compared to CE, BCE can more quickly lead to NC on the training dataset and achieves better feature compactness and distinctiveness, resulting in higher classification performance on the test dataset.

# Getting Started
## Requirements
* CUDA
* Python
* torch
* torchvision
* numpy

## Preparing Datasets

# Training and Testing

# Results
## 1. Experiments using ViT on CIFAR10
![Distribution of decision scores and biases for ViTs with varying initial mean on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_mean.png)
<br />

![Distribution of decision scores and biases for ViTs with varying weight decay on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar10_bias_weight_decay.png)
<br />

## 2. Experiments using ViT on CIFAR100
![Distribution of decision scores and biases for ViTs with varying initial mean on CIFAR100](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar100_bias_mean.png)
<br />

![Distribution of decision scores and biases for ViTs with varying weight decay on CIFAR100](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ViT_cifar100_bias_weight_decay.png)
<br />

## 3. Experiments using Swin Transformer on CIFAR10
![Distribution of decision scores and biases for Swin Transformers with varying initial mean on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar10_bias_mean.png)
<br />

![Distribution of decision scores and biases for Swin Transformers with varying weight decay on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar10_bias_weight_decay.png)
<br />

## 4. Experiments using Swin Transformer on CIFAR100
![Distribution of decision scores and biases for Swin Transformers with varying initial mean on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar100_bias_mean.png)
<br />

![Distribution of decision scores and biases for Swin Transformers with varying weight decay on CIFAR10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/Swin_cifar100_bias_weight_decay.png)
<br />

## 5. Visualization using t-SNE
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_CIFAR10.gif)

### 5.1. Display for specific training epochs
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=6](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch6.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=7](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch7.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=8](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch8.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=9](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch9.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=10](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch10.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=11](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch11.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=12](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch12.png)
![Visualize the features learned by ResNet18 on CIFAR-10 using t-SNE, epoch=13](https://github.com/Xiao-hb/BCE-vs.-CE/blob/main/Figs/ResNet18_cifar10_epoch13.png)

# Citation
