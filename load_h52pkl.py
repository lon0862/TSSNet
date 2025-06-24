import os
import h5py
import numpy as np
from argparse import ArgumentParser
from utils.save_utils import eval_store_models

def transform_h52pkl(args, h5_file_name):
    file = h5py.File(h5_file_name, 'r')
    for key in file.keys():
        if args.model_name == 'LaneGCN':
            data = file[key][:].reshape(-1, 6, 30, 3) # [B, K, F, 3]
        else:
            data = file[key][:].reshape(-1, 6, 30, 4) # [B, K, F, 4]
        scenario_id = []
        init_agent = []
        for i in range(len(data)):
            seq_id = int(data[i,0,0,0])
            scenario_id.append(seq_id)
            init_agent.append(data[i,:,:,1:3])
        init_agent = np.array(init_agent)

    print("len of data:", init_agent.shape)
    if args.data_name == 'itri':
        pred_path = os.path.join(args.pred_path_root, f'pkl_{args.split_type}', args.obj_type, args.split, args.model_name)
    elif args.data_name == 'av1':
        pred_path = os.path.join(args.pred_path_root, 'pkl', args.split, args.model_name)

    os.makedirs(pred_path, exist_ok=True)
    eval_store_models(scenario_id, pred_path, init_agent)
    print("finish saving for pkl files")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_path_root', type=str, default='pred_results_models_itri')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--model_name', type=str, default='HPNet', choices=['HPNet', 'DGFNet', 'SmartRefine', 'SIMPL'])
    parser.add_argument('--data_name', type=str, default='av1', choices=['av1', 'itri'])
    parser.add_argument('--split_type', type=str, default='split', choices=['split', 'continuous'])
    parser.add_argument('--obj_type', type=str, default='bimo', choices=['bimo', 'bus', 'car', 'pedestrian', 'truck'])
    args = parser.parse_args()

    if args.data_name == 'itri':
        h5_file_name = os.path.join(args.pred_path_root, f'h5_{args.split_type}', args.obj_type, args.split, args.model_name, 'submission.h5')
    elif args.data_name == 'av1':
        h5_file_name = os.path.join(args.pred_path_root, 'h5', args.split, args.model_name, 'submission.h5')
    
    transform_h52pkl(args, h5_file_name)