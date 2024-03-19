import os
import numpy as np
import math
from os import listdir
from os.path import isfile, join

users = [
    [],         # dir = 0（不存在）
    [1, 4, 9],  # 目录 = 1
    [4, 10, 13],  # 目录 = 2
    [16, 19, 21],  # 目录 = 3
    [1, 2, 12],  # 目录 = 4
    [6, 10, 11],  # 目录 = 5
    [2, 5, 13],  # 目录 = 6
    [15, 18, 21],  # 目录 = 7
    [2, 7, 9],  # 目录 = 8
    [16, 17, 18],  # 目录 = 9
    [1, 8, 11],  # 目录 = 10
    [2, 7, 13],  # 目录 = 11
    [1, 2, 9],  # 目录 = 12
    [1, 2, 9],  # 目录 = 13
    [10, 17, 22],  # 目录 = 14
    [1, 2, 7],  # 目录 = 15
    [1, 2, 13],  # 目录 = 16
    [3, 5, 12],  # 目录 = 17
    [2, 7, 11],  # 目录 = 18
    [2, 3, 6],  # 目录 = 19
    [3, 4, 7],  # 目录 = 20
    [4, 11, 12],  # 目录 = 21
    [1, 2, 17],  # 目录 = 22
]

gt_path = './LPW'
result_path = './result_files'

def ComputeStatForOneFile(dir, maxDistance):
    result = np.zeros(maxDistance)
    noPupils = 0

    for user in users[dir]:
        # 读取 ground truth 文件
        gtFile = f"{gt_path}/{dir}/{user}.txt"
        contents = open(gtFile, "r").read().split("\n")
        gt = []
        for line in contents:
            if line == "":
                break

            x, y = line.split(' ')
            gt.append((float(x), float(y)))

        # 读取预测结果文件
        predFile = f"{result_path}/{dir}/{user}.txt"
        contents = open(predFile, "r").read().split("\n")
        pred = []
        for line in contents:
            if line == "":
                break

            x, y = line.split(' ')
            pred.append((float(x), float(y)))

        if len(pred) < len(gt):
            print(f"预测结果长度 {len(pred)} < ground truth 长度 {len(gt)}!")
            os._exit(0)

        # 计算检测准确率随距离增加的情况
        noPupils += len(pred)
        for index in range(len(pred)):
            x1, y1 = gt[index]
            x2, y2 = pred[index]

            dx = x1 - x2
            dy = y1 - y2
            distance = math.sqrt(dx*dx + dy*dy)
            for d in range(maxDistance):
                if distance <= d:
                    result[d] += 1

    return noPupils, result

#---------------------------------------------------------------------
# 主程序
#
if __name__ == '__main__':
    maxDistance = 15+1

    results = []
    totalNoImgs = 0

    for dir in range(1, 23, 1):
        noImgs, result = ComputeStatForOneFile(dir, maxDistance)
        results.append((noImgs, result));
        totalNoImgs += noImgs
        print(f"{100*result[5]/noImgs:6.2f}")

    sum = np.zeros(maxDistance)
    for index in range(maxDistance):
        for j in range(len(results)):
            sum[index] += results[j][1][index]                   # 加权平均

    for index in range(maxDistance):
        print(f"{index:3d} {100*sum[index]/totalNoImgs:6.2f}")   # 加权平均
