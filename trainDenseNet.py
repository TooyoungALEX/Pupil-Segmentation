from __future__ import print_function

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torchvision.utils import save_image
from PIL import Image

import cv2
import warnings
import math

from densenet import DenseNet2D
from utils import CrossEntropyLoss2d, GeneralizedDiceLoss, get_predictions
from ellipseFitError import EllipseFitErrorDenseNet

CE = CrossEntropyLoss2d()
DICE = GeneralizedDiceLoss(softmax=True, reduction=True)

# ----------------------- _parse_image_function ----------------------
feature = {'train/image': tf.io.FixedLenFeature([], tf.string),
           'train/label': tf.io.FixedLenFeature([], tf.string)}


# 解析输入的 tf.Example proto，使用上面的字典。
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature)


# ----------------------- processData ----------------------
# 给定一个模型和数据集，使用处理模式在模型上处理数据集
#
def processData(model, dataSet, epoch, processingMode="train"):
    global imageNo
    imageNo = 1

    # 按 batchSize 对数据进行批处理
    dataSetByBatch = dataSet.batch(batch_size=batchSize)

    noImages = 0
    current_batch = 0
    total_loss = 0
    interval_loss = 0
    for image_features in dataSetByBatch:
        image = tf.io.decode_raw(image_features['train/image'], tf.uint8)
        label = tf.io.decode_raw(image_features['train/label'], tf.uint8)

        image = tf.reshape(image, [-1, Size_Y, Size_X])
        image = tf.expand_dims(image, 1)

        label = tf.reshape(label, [-1, Size_Y, Size_X])

        noImages += len(image)
        current_batch += 1

        # 将图像转换为 [0-1] 并加载到 GPU 上
        image = image.numpy().astype(np.float32) / 255
        label = label.numpy().astype(np.uint8) / 255

        image = torch.from_numpy(image)
        image = image.to(device)

        label = torch.from_numpy(label)
        label = label.to(device).long()

        # 将图像通过模型，然后通过 Sigmoid 函数
        output = model(image)

        # 计算损失：交叉熵 + DICE
        loss = CE(output, label)

        target = label.to(device).long()
        loss += DICE(output, target)

        # 添加椭圆拟合误差正则化项
        predict = get_predictions(output)
        ellipseFitError = EllipseFitErrorDenseNet(predict, label) * 0.1
        print(ellipseFitError.item())
        loss += ellipseFitError

        if processingMode == "train":
            # 在训练模式下进行反向传播并更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss)
        interval_loss += float(loss)

        # 在屏幕上打印信息
        DIV = 100
        if current_batch % DIV == 0:
            avg_loss = total_loss / current_batch
            avg_interval_loss = interval_loss / DIV
            interval_loss = 0
            print(
                f"Epoch: {epoch}, Batch: {current_batch:4}, avg. loss: {avg_loss:.5f}, int. loss: {avg_interval_loss:.5f}")

            if processingMode == "train":
                fileno = int(current_batch / DIV)

                predict = get_predictions(output)  # 提取结果
                im = predict[0].cpu().numpy()
                im[im >= 1] = 255
                im = Image.fromarray(np.uint8(im))
                im.save(f"output/{fileno:05d}.png")

        del image, label, output, loss

    return total_loss / current_batch  # 返回整个 epoch 的平均损失


# -------------------------- main function -------------------------------
def main():
    # 将以下变量设置为全局变量，以便其他函数可以访问它们
    global Size_X, Size_Y, batchSize, device, optimizer

    torch.manual_seed(1)
    warnings.filterwarnings("ignore")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("DEVICE: ", device)

    shuffle_buffer = int(2000)
    Size_X, Size_Y, batchSize = 320, 240, 16
    # Size_X, Size_Y, batchSize = 640, 480, 8

    # 数据集
    rawTrainingDataset = tf.data.TFRecordDataset(f"./tfRecords/{Size_X}x{Size_Y}-train")
    rawValidationDataset = tf.data.TFRecordDataset(f"./tfRecords/{Size_X}x{Size_Y}-validation")

    trainingSet = rawTrainingDataset.map(_parse_image_function)
    validationSet = rawValidationDataset.map(_parse_image_function)

    # 将训练好的模型保存在每个 epoch 结束时的文件夹中
    trained_model_path = './trained_model'
    if not os.path.exists(trained_model_path):
        os.mkdir('./trained_model')

    # 将调试输出图像保存在 output 文件夹中
    output_path = './output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 创建一个 DenseNet 模型
    learning_rate = 1e-3
    model = DenseNet2D(in_channels=1, out_channels=2, dropout=True, prob=0.2)

    # 加载之前保存的模型（如果存在）
    # model_file_name = "trained_model/model1.pt"
    # model.load_state_dict(torch.load(model_file_name))

    model.to(device)  # 将模型上传到 GPU

    # 创建一个优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 打开日志文件
    logFile = open("log.txt", "w")

    noEpochs = 50
    for epoch in range(1, noEpochs + 1, 1):
        print(f"==============Training Epoch: {epoch}===============")

        # ----------------- 训练 ------------------------
        # 打乱训练数据并进行批处理
        # 将模型置于训练模式
        model.train()
        torch.set_grad_enabled(True)  # 显式启用梯度计算

        trainingSet = trainingSet.shuffle(buffer_size=shuffle_buffer)
        avg_training_loss = processData(model, trainingSet, epoch, "train")
        print(f"----> End of training epoch {epoch:2d}. avg. loss: {avg_training_loss:.5f}")

        # ----------------- 验证 ------------------------
        # 将模型置于评估模式并进行验证
        model.eval()
        torch.set_grad_enabled(False)  # 显式禁用梯度计算

        avg_val_loss = processData(model, validationSet, epoch, "validation")
        print(f"----> End of validation. epoch {epoch:2d}. avg. loss: {avg_val_loss:.5f}")
        scheduler.step(avg_val_loss)

        # 将 epoch 的平均损失写入日志文件
        logFile.write(f"{epoch:4d}{avg_training_loss:10.5f}{avg_val_loss:10.5f}\n")
        logFile.flush()

    # 将训练好的模型保存到磁盘
    savename = f"{trained_model_path}/DenseNet-trained-model.pt"
    torch.save(model.state_dict(), savename)

    logFile.close()


# -------------------------------------------------------
# 调用主函数并开始处理
if __name__ == '__main__':
    main()
