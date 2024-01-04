from model_wrapper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C, MLP_Wrapper_D, MLP_Wrapper_E, MLP_Wrapper_F, FCN_Wrapper
from utils import read_json
import numpy as np
import torch
from dataloader import CNN_Data, FCN_Data, MLP_Data, MLP_Data_apoe, CNN_MLP_Data
from torch.utils.data import Dataset, DataLoader
from utils import matrix_sum, get_MCC, get_confusion_matrix, write_raw_score, DPM_statistics, timeit, read_csv

def transform(i):
    if i == 'nl':
        return 0
    elif i == 'scd':
        return 1
    elif i == 'mci':
        return 2
    else:
        return 3

def getMlp(config):
    mlp_setting = config

    mlp1, mlp2, mlp3, mlp4 = None, None, None, None
    mlp = [mlp1, mlp2, mlp3, mlp4]
    all_type = [['NL', 'NonNL'], ['SCD', 'NonSCD'], ['MCI', 'NonMCI'], ['AD', 'NonAD']]
    number = [14, 220, 22, 299]
    for i in range(len(mlp)):
        mlp[i] = MLP_Wrapper_A(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=0,
                            seed=seed,
                            model_name='mlp_A',
                            metric='accuracy', type1=all_type[i][0], type2=all_type[i][1])
        mlp[i].load_model(number[i])
    return mlp

def get_accu(matrix):
    # return float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))
    return float(matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3])/ float(sum(matrix[0]) + sum(matrix[1]) + sum(matrix[2]) + sum(matrix[3]))

def getFcn(config):
    fcn_setting = config['fcn']            
    fcn1, fcn2, fcn3, fcn4= None, None, None, None
    fcn = [fcn1, fcn2, fcn3, fcn4]
    all_type = [['NL', 'NonNL'], ['SCD', 'NonSCD'], ['MCI', 'NonMCI'], ['AD', 'NonAD']]
    number = [3560, 660, 720, 2780]
    for i in range(len(fcn)):
        fcn[i] = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        exp_idx         = 0,
                        seed            = seed,
                        model_name      = 'fcn',
                        metric          = 'accuracy',
                        type1=all_type[i][0], type2=all_type[i][1]
                        )
        fcn[i].load_model(number[i])
    return fcn

def generate_DPM(config):
    fcn_setting = config['fcn']
    fcn = getFcn(config)  
    with torch.no_grad():
        for stage in ['ABCD']:
            Data_dir = fcn_setting['Data_dir']
            data = FCN_Data(Data_dir, 0, stage=stage, whole_volume=True, seed=seed, patch_size=fcn_setting['patch_size'], type1=None, type2=None)
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            for idx, (inputs, labels) in enumerate(dataloader):
                # generate DPM
                for i in range(4):
                    fcn_model = fcn[i]

                    fcn_model.generate_DPM(inputs, data.Data_list[idx])

def mlp_predict(config):
    mlp_setting = config['mlp_A']
    mlp = getMlp(mlp_setting)
    all_type = [['NL', 'NonNL'], ['SCD', 'NonSCD'], ['MCI', 'NonMCI'], ['AD', 'NonAD']]
    with torch.no_grad():
        for stage in ['ABCD']:
            print(stage)
            result = [[], [], [], []]
            for i in range(len(mlp)):
                model = mlp[i]
                type1, type2 = all_type[i][0], all_type[i][1]
                data = MLP_Data('./DPMs_{}_{}/fcn_exp0/'.format(type1, type2), 0, stage=stage, roi_threshold=mlp_setting['roi_threshold'], roi_count=mlp_setting['roi_count'], choice=mlp_setting['choice'], seed=seed, type1=type1, type2=type2)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)

                for idx, (inputs, labels, _) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    res = model.predict(inputs)
                    # to edit the weight
                    if i == 3:
                        # res = [x*1.1 if x > 0.5 else x for x in res]
                        result[i].append([res, labels])
                    else:
                        result[i].append([res, labels])
            mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            for i in range(len(result[0])):
                labels = result[0][i][1] # labels.length
                tmp = []
                for _ in range(len(labels)):
                    tmp.append([])
                for x in range(4):
                    res = result[x][i][0]
                    for num in range(len(res)):
                        dic = tmp[num]
                        dic.append(res)
                max_type =  [d.index(max(d)) for d in tmp]
            #     # fake
            #     preds = []
            #     for ss in range(len(max_type)):
            #         type = max_type[ss]
            #         count = 0
            #         for value in tmp[ss].values():
            #             if value == 2:
            #                 count += 1
            #         if tmp[ss][type] == 2 and count > 1:
            #             preds.append(labels[ss])
            #         else:
            #             preds.append(transform(type))
            #     # fake
                preds = max_type
                for i in range(len(labels)):
                    mat[labels[i]][preds[i]] += 1
            print(get_accu(mat))
            print(mat)
            


if __name__ == "__main__":
    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']

    generate_DPM(config)
    mlp_predict(config)