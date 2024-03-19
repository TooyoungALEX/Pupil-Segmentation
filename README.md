该论文提出了一种使用椭圆拟合误差作为形状先验正则化项的方法，该项可以添加到像素级损失函数（例如二元交叉熵）中，以训练卷积神经网络（CNN）进行瞳孔分割。作者通过训练一种轻量级的UNet架构，并使用三个广泛使用的真实世界数据集（ExCuSe、ElSe和LPW）来评估所提方法的性能，这三个数据集共包含约23万张图像。实验结果表明，所提出的方法在所有数据集上均获得了已知的最佳瞳孔检测率。

如果您使用了本文提供的代码，请引用该论文： Accurate CNN-based Pupil Segmentation with an Ellipse Fit Error Regularization Term. Expert Systems with Applications

您可以从以下链接下载完整的LPW数据集：http://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip

LPW数据集的GT分割地图可以从以下链接下载：https://www.kaggle.com/cuneytakinlar/lpw-gt-segmentation-maps

在运行代码之前，请安装所需的Python包。您可以创建一个虚拟环境（使用conda），然后在激活该虚拟环境后运行以下命令来下载和安装所需的包：

```
conda install tensorflow opencv torchvision numpy cudatoolkit -c pytorch
```

在运行代码之前，您需要选择用于数据集创建、训练和测试的图像大小。在每个文件中，都有一行需要更改的代码。默认情况下，这些值设置为width=320，height=240。您可以将它们设置为任何您想要的值。

运行代码的步骤如下：

1. 创建训练和验证数据集：
```
python createTFData.py
```

2. 训练模型。有两个可以使用的语义分割模型：UNet和DenseNet。您可以选择任何一个进行训练。训练过程中，每个周期后，当前模型文件都会保存在trained_model目录下。
```
python trainUNet.py
```

3. 测试模型。编辑testUNet.py并输入要使用的模型文件的名称，然后运行：
```
python testUNet.py
```

4. 计算瞳孔检测率：
```
python computePupilDetectionStats.py
```

或者，您也可以使用DenseNet进行训练和测试。在这种情况下，您需要使用trainDenseNet.py和testDenseNet.py。