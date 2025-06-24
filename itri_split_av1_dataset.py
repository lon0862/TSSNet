from tqdm import tqdm
from argparse import ArgumentParser
import os
import pandas as pd

import json
import torch
import numpy as np
import shutil
from sklearn.utils import shuffle

def compute_agent_fut_traj_length(df, num_historical_steps):
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
    agent_df = agent_df.sort_values('TIMESTAMP')
    agent_pos_np = agent_df[['X', 'Y']].to_numpy().astype(float)
    agent_pos = torch.from_numpy(agent_pos_np).float()
    fut_pos = agent_pos[num_historical_steps-1:]
    fut_vec = fut_pos[1:] - fut_pos[:-1]
    fut_length = torch.norm(fut_vec, dim=0).sum().item()

    return fut_length

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='../ITRI_Dataset')
    parser.add_argument('--num_historical_steps', type=int, default=20)
    parser.add_argument('--split_type', type=str, default='continuous', choices=['split', 'continuous'])
    parser.add_argument('--object_type', type=str, default='car', \
                        choices=['bimo', 'bus', 'car', 'pedestrian', 'truck'])
    args = parser.parse_args()

    city_name = 'ITRI'
    av1_root = os.path.join(args.root, 'av1_dataset_' + args.split_type, args.object_type)
    all_files_path = os.path.join(av1_root, 'all/data')
    train_dataset_path = os.path.join(av1_root, 'train/data')
    val_dataset_path = os.path.join(av1_root, 'val/data')
    os.makedirs(train_dataset_path, exist_ok=True)
    os.makedirs(val_dataset_path, exist_ok=True)

    file_lengths = []
    for file in tqdm(sorted(os.listdir(all_files_path))):
        file_path = os.path.join(all_files_path, file)
        df = pd.read_csv(file_path)
        traj_len = compute_agent_fut_traj_length(df, args.num_historical_steps)
        file_lengths.append((file, traj_len))

    lengths = [l for _, l in file_lengths]
    min_len, max_len = min(lengths), max(lengths)
    bins = np.linspace(min_len, max_len+0.001, 11)

    bin_lists = [[] for _ in range(10)]
    for file, length in file_lengths:
        bin_idx = np.digitize(length, bins, right=False) - 1
        bin_idx = min(bin_idx, 9) 
        bin_lists[bin_idx].append(file)

    # Assign files to train / valid
    for bin_list in bin_lists:
        shuffled = shuffle(bin_list, random_state=42)
        split_idx = len(shuffled) // 2
        train_files = shuffled[:split_idx]
        valid_files = shuffled[split_idx:]

        for f in train_files:
            shutil.copy(os.path.join(all_files_path, f), os.path.join(train_dataset_path, f))
        for f in valid_files:
            shutil.copy(os.path.join(all_files_path, f), os.path.join(val_dataset_path, f))
