from model_wrapper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C, MLP_Wrapper_D, MLP_Wrapper_E, MLP_Wrapper_F
from utils import read_json
import numpy as np
import torch

def six_model_test(config):
    mlp_setting = config

    mlp1, mlp2, mlp3, mlp4, mlp5, mlp6 = '', '', '', '', '', ''
    mlp = [mlp1, mlp2, mlp3, mlp4, mlp5, mlp6]
    all_type = [['nl', 'scd'], ['nl', 'mci'], ['nl', 'ad'], ['scd', 'mci'], ['scd', 'ad'], ['mci', 'ad']]
    number = [100, 291, 145, 238, 293, 14]
    for i in range(6):
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

    with torch.no_grad():
         for stage in ['train', 'valid', 'test']:


if __name__ == "__main__":
    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']
    six_model_test(config["mlp_A"])