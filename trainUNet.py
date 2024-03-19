from __future__ import print_function

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf
from torchvision.utils import save_image

import cv2
import warnings
import math

from unet import UNet, MyUNet
from ellipseFitError import EllipseFitErrorUNet

#----------------------- _parse_image_function ----------------------
# 定义解析 TFRecord 的函数
feature = {'train/image': tf.io.FixedLenFeature([], tf.string),
           'train/label': tf.io.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature)

#----------------------- processData ----------------------
# 给定模型和数据集，使用处理模式处理数据集
def processData(model, dataSet, epoch, processingMode="train"):
    global imageNo
    imageNo = 1

    # 按批次访问数据
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
        label = tf.expand_dims(label, 1)

        noImages += len(image)
        current_batch += 1

        # 将图像转换为[0-1]并加载到GPU
        image = image.numpy().astype(np.float32)/255
        label = label.numpy().astype(np.float32)/255

        image = torch.from_numpy(image)
        image = image.to(device)

        label = torch.from_numpy(label)
        label = label.to(device)

        # 将图像通过模型并经过 sigmoid 处理
        output = model(image)
        output = nn.functional.sigmoid(output)

        # 计算损失
        loss = F.binary_cross_entropy(output, label)

        # 添加椭圆拟合误差正则化项
        ellipseFitError = EllipseFitErrorUNet(output, label) * 0.1
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
            avg_loss = total_loss/current_batch
            avg_interval_loss = interval_loss/DIV
            interval_loss = 0
            print(f"Epoch: {epoch}, Batch: {current_batch:4}, avg. loss: {avg_loss:.5f}, int. loss: {avg_interval_loss:.5f}")

            if processingMode == "train":
                fileno = int(current_batch/DIV)
                filename = f"output/{fileno:05d}.png"
                save_image(output[0], filename)

        del image, label, output, loss

    return total_loss/current_batch # 返回 epoch 中的平均损失

#-------------------------- main function -------------------------------
def main():
    global Size_X, Size_Y, batchSize, device, optimizer

    torch.manual_seed(1)
    warnings.filterwarnings("ignore")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("DEVICE: ", device)

    shuffle_buffer = int(2000)
    Size_X, Size_Y, batchSize = 320, 240, 16

    # 加载训练和验证数据集
    rawTrainingDataset = tf.data.TFRecordDataset(f"./tfRecords/{Size_X}x{Size_Y}-train")
    rawValidationDataset = tf.data.TFRecordDataset(f"./tfRecords/{Size_X}x{Size_Y}-validation")

    trainingSet = rawTrainingDataset.map(_parse_image_function)
    validationSet = rawValidationDataset.map(_parse_image_function)

    trained_model_path = './trained_model'
    if not os.path.exists(trained_model_path): os.mkdir('./trained_model')

    output_path = './output'
    if not os.path.exists(output_path): os.mkdir(output_path)

    learning_rate = 1e-4
    model = MyUNet(32)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    logFile = open("log.txt", "w")

    noEpochs = 30
    for epoch in range(1, noEpochs+1, 1):
        print(f"==============Training Epoch: {epoch}===============")

        model.train()
        torch.set_grad_enabled(True)

        trainingSet = trainingSet.shuffle(buffer_size=shuffle_buffer)
        avg_training_loss = processData(model, trainingSet, epoch, "train")
        print(f"----> End of training epoch {epoch:2d}. avg. loss: {avg_training_loss:.5f}")

        model.eval()
        torch.set_grad_enabled(False)

        avg_val_loss = processData(model, validationSet, epoch, "validation")
        print(f"----> End of validation. epoch {epoch:2d}. avg. loss: {avg_val_loss:.5f}")
        scheduler.step(avg_val_loss)

        logFile.write(f"{epoch:4d}{avg_training_loss:10.5f}{avg_val_loss:10.5f}\n")
        logFile.flush()

    savename = f"{trained_model_path}/Unet-trained-model.pt"
    torch.save(model.state_dict(), savename)

    logFile.close()

if __name__ == '__main__':
    main()
