import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import random
import json
import csv
import random
import os
import time


class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def random_sample(self, volume):
        """sample random patch from numpy array data"""
        X, Y, Z = volume.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        return volume[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+47, y:y+47, z:z+47]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches


def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List


def padding(tensor, win_size=23):
    A = np.ones((tensor.shape[0]+2*win_size, tensor.shape[1]+2*win_size, tensor.shape[2]+2*win_size)) * tensor[-1,-1,-1]
    A[win_size:win_size+tensor.shape[0], win_size:win_size+tensor.shape[1], win_size:win_size+tensor.shape[2]] = tensor
    return A.astype(np.float32)


def get_confusion_matrix(preds, labels):
    labels = labels.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            # matrix[0][labels[index]] += 1
            matrix[labels[index]][0] += 1
        elif np.amax(pred) == pred[1]:
            # matrix[1][labels[index]] += 1
            matrix[labels[index]][1] += 1
        elif np.amax(pred) == pred[2]:
            # matrix[2][labels[index]] += 1
            matrix[labels[index]][2] += 1
        elif np.amax(pred) == pred[3]:
            # matrix[3][labels[index]] += 1
            matrix[labels[index]][3] += 1
    return matrix


def matrix_sum(A, B): 
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2], A[0][3]+B[0][3]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2], A[1][3]+B[1][3]],
            [A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2], A[2][3]+B[2][3]],
            [A[3][0]+B[3][0], A[3][1]+B[3][1], A[3][2]+B[3][2], A[3][3]+B[3][3]]]


def get_accu(matrix):
    # return float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))
    return float(matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3])/ float(sum(matrix[0]) + sum(matrix[1]) + sum(matrix[2]) + sum(matrix[3]))


def get_MCC(matrix):
    TypeError
    TP, TN, FP, FN = float(matrix[0][0]), float(matrix[1][1]), float(matrix[0][1]), float(matrix[1][0])
    upper = TP * TN - FP * FN
    lower = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return upper / (lower**0.5 + 0.000000001)


def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    v = np.exp(x1) + np.exp(x2)
    risk = np.exp(x2) / v
    return risk

def get_AD_risk_new(raw):
    x1, x2, x3, x4 = raw[0, :, :, :], raw[1, :, :, :], raw[2, :, :, :], raw[3, :, :, :]
    v = np.exp(x1) + np.exp(x2) + np.exp(x3) + np.exp(x4)
    a, b, c, d = np.exp(x1) / v, np.exp(x2) / v, np.exp(x3) / v, np.exp(x4) / v
    risk = np.stack([a, b, c, d], axis=3)
    return risk


def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def write_raw_score_sk(f, preds, labels):
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = []
    for a in your_list[1:]:
        if a[1] == 'NL':
            labels.append(0)
        elif a[1] == 'SCD':
            labels.append(1)
        elif a[1] == 'MCI':
            labels.append(2)
        else:
            labels.append(3)
    return filenames, labels


def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list:
        try:
            demor = list(map(float, line[2:5]))
            gender = [0, 1] if demor[1] == 1 else [1, 0]
            demor = [(demor[0]-70.0)/10.0] + gender + [(demor[2]-27)/2]
            # demor = [demor[0]] + gender + demor[2:]
        except:
            continue
        filenames.append(line[0])
        if line[1] == 'NL':
            label = 0
        elif line[1] == 'SCD':
            label = 1
        elif line[1] == 'MCI':
            label = 2
        else:
            label = 3
        labels.append(label)
        demors.append(demor)
    return filenames, labels, demors


def read_csv_complete_apoe(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list:
        try:
            demor = list(map(float, line[2:6]))
            gender = [0, 1] if demor[1] == 1 else [1, 0]
            demor = [(demor[0] - 70.0) / 10.0] + gender + [(demor[2] - 27) / 2] + [demor[3]]
        except:
            continue
        filenames.append(line[0])
        if line[1] == 'NL':
            label = 0
        elif line[1] == 'SCD':
            label = 1
        elif line[1] == 'MCI':
            label = 2
        else:
            label = 3
        labels.append(label)
        demors.append(demor)
    return filenames, labels, demors


def data_split(repe_time):
    with open('./lookupcsv/ABCD.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    # labels, train_valid, test = your_list[0:1], your_list[1:338], your_list[338:]
    labels = your_list[0:1]
    tmp = your_list[1:]
    train_valid, test = tmp[0:416], tmp[416:]
    for i in range(repe_time):
        random.shuffle(train_valid)
        folder = 'lookupcsv/exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder) 
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid[:320])
        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid[320:])
        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + test)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


# def DPM_statistics(DPMs, Labels):
#     shape = DPMs[0].shape[1:]
#     voxel_number = shape[0] * shape[1] * shape[2]
#     TP, FP, TN, FN = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
#     for label, DPM in zip(Labels, DPMs):
#         risk_map = get_AD_risk(DPM)
#         if label == 0:
#             TN += (risk_map < 0.5).astype(int)
#             FP += (risk_map >= 0.5).astype(int)
#         elif label == 1:
#             TP += (risk_map >= 0.5).astype(int)
#             FN += (risk_map < 0.5).astype(int)
#     tn = float("{0:.2f}".format(np.sum(TN) / voxel_number))
#     fn = float("{0:.2f}".format(np.sum(FN) / voxel_number))
#     tp = float("{0:.2f}".format(np.sum(TP) / voxel_number))
#     fp = float("{0:.2f}".format(np.sum(FP) / voxel_number))
#     matrix = [[tn, fn], [fp, tp]]
#     count = len(Labels)
#     TP, TN, FP, FN = TP.astype(float)/count, TN.astype(float)/count, FP.astype(float)/count, FN.astype(float)/count
#     ACCU = TP + TN
#     F1 = 2*TP/(2*TP+FP+FN)
#     MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))
#     return matrix, ACCU, F1, MCC

def maxxx(r1, r2, r3):
    return np.maximum(np.maximum(r1, r2), r3)

def DPM_statistics(DPMs, Labels):
    shape = DPMs[0].shape[1:]
    voxel_number = shape[0] * shape[1] * shape[2]

    shapeOfMatrixHandler = (4, 4, shape[0], shape[1], shape[2])
    matrixHandler = np.zeros(shapeOfMatrixHandler)
    for label, DPM in zip(Labels, DPMs):
        risk_map = get_AD_risk(DPM)
        r1, r2, r3, r4 = risk_map[0], risk_map[1], risk_map[2], risk_map[3]
        if label == 0:
            matrixHandler[0][0] += (r1 >= maxxx(r2, r3, r4)).astype(int)
            matrixHandler[0][1] += (r2 >= maxxx(r1, r3, r4)).astype(int)
            matrixHandler[0][2] += (r3 >= maxxx(r2, r1, r4)).astype(int)
            matrixHandler[0][3] += (r4 >= maxxx(r2, r3, r1)).astype(int)
        elif label == 1:
            matrixHandler[1][0] += (r1 >= maxxx(r2, r3, r4)).astype(int)
            matrixHandler[1][1] += (r2 >= maxxx(r1, r3, r4)).astype(int)
            matrixHandler[1][2] += (r3 >= maxxx(r2, r1, r4)).astype(int)
            matrixHandler[1][3] += (r4 >= maxxx(r2, r3, r1)).astype(int)
        elif label == 2:
            matrixHandler[2][0] += (r1 >= maxxx(r2, r3, r4)).astype(int)
            matrixHandler[2][1] += (r2 >= maxxx(r1, r3, r4)).astype(int)
            matrixHandler[2][2] += (r3 >= maxxx(r2, r1, r4)).astype(int)
            matrixHandler[2][3] += (r4 >= maxxx(r2, r3, r1)).astype(int)
        elif label == 3:
            matrixHandler[3][0] += (r1 >= maxxx(r2, r3, r4)).astype(int)
            matrixHandler[3][1] += (r2 >= maxxx(r1, r3, r4)).astype(int)
            matrixHandler[3][2] += (r3 >= maxxx(r2, r1, r4)).astype(int)
            matrixHandler[3][3] += (r4 >= maxxx(r2, r3, r1)).astype(int)
        else:
            print('wrong')
    
    matrix = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            matrix[i][j] = float("{0:.2f}".format(np.sum(matrixHandler[i][j]) / voxel_number))
    count = len(Labels)
    # TP, TN, FP, FN = TP.astype(float)/count, TN.astype(float)/count, FP.astype(float)/count, FN.astype(float)/count
    # ACCU = TP + TN
    # F1 = 2*TP/(2*TP+FP+FN)
    # MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))

    ACCU = (matrixHandler[0][0] + matrixHandler[1][1] + matrixHandler[2][2] + matrixHandler[3][3]).astype(float) / count

    # accu = [0,0,0,0]
    # column = [0,0,0,0]
    # line = [0,0,0,0]
    # ACCU = 0
    # recall = 0
    # precision = 0
    # for i in range(0,4):
    #     accu[i] = matrix[i][i]
    # for i in range(0,4):
    #     for j in range(0,4):
    #         column[i]+=matrix[j][i]
    # for i in range(0,4):
    #     for j in range(0,4):
    #         line[i]+=matrix[i][j]
    # for i in range(0,4):
    #     ACCU += float(accu[i])/count
    # for i in range(0,4):
    #     if column[i] != 0:
    #         recall+=float(accu[i])/column[i]
    # recall = recall / 4
    # for i in range(0,4):
    #     if line[i] != 0:
    #         precision+=float(accu[i])/line[i]
    # precision = precision / 4
    # F1 = (2 * (precision * recall)) / (precision + recall)
    return matrix, ACCU


