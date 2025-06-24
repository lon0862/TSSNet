from tqdm import tqdm
from argparse import ArgumentParser
import os
import pandas as pd

import json
import torch

def get_agent_and_save(av_df, others_df, obj_type, save_root, seq_id, valid_thres, ts_len=50):
    type_df = others_df[others_df['OBJECT_TYPE'] == obj_type]
    
    # count track_id's timestamp number
    track_ts_count = type_df.groupby('TRACK_ID')['TIMESTAMP'].nunique()

    # find has 50 ts one as 'AGENT'
    candidate_ids = track_ts_count[track_ts_count == ts_len].index.tolist()

    if candidate_ids:
        for chosen_id in candidate_ids:
            others_df_copy = others_df.copy()

            others_df_copy.loc[others_df_copy['TRACK_ID'] == chosen_id, 'OBJECT_TYPE'] = 'AGENT'
            others_df_copy.loc[others_df_copy['TRACK_ID'] != chosen_id, 'OBJECT_TYPE'] = 'OTHERS'
            df = pd.concat([av_df, others_df_copy], ignore_index=True)
            df = df.sort_values(by='TIMESTAMP')
            df.to_csv(f"{save_root}/{seq_id}.csv", index=False)
            seq_id +=1

    else:
        assert "no target AGENT"

    
    return seq_id


def split_by_ts(av_df, others_df, unique_ts, seq_id, obj_type, save_root, split_type, valid_thres, max_len=50):
    # groupby 提取每個 TRACK_ID 的 timestamp
    type_df = others_df[others_df['OBJECT_TYPE'] == obj_type]
    track_timestamps = type_df.groupby('TRACK_ID')['TIMESTAMP'].apply(list).to_dict()

    max_ts = len(unique_ts)
    seg_start = 0
    while (seg_start + max_len - 1) <= max_ts:
        seg_end = seg_start + max_len - 1
        found = False

        for track_id, ts_list in track_timestamps.items():
            ts_start = ts_list[0]
            ts_end = ts_list[-1]
            if seg_start >= ts_start and seg_end <= ts_end:
                found = True
                break
            if found:
                break
        
        if found:
            others_df_sliced = others_df[(others_df['TIMESTAMP'] >= seg_start) & (others_df['TIMESTAMP'] <= seg_end)]
            av_df_sliced = av_df[(av_df['TIMESTAMP'] >= seg_start) & (av_df['TIMESTAMP'] <= seg_end)]
            seq_id = get_agent_and_save(av_df_sliced, others_df_sliced, \
                                            obj_type, save_root, seq_id, valid_thres, ts_len=max_len)

            if split_type == 'continuous':
                seg_start += 1 # continuous split
            else:
                seg_start += max_len # direct split 
            
        else:
            seg_start +=1

    return seq_id

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='../ITRI_Dataset')
    parser.add_argument('--num_historical_steps', type=int, default=20)
    parser.add_argument('--num_future_steps', type=int, default=30)
    parser.add_argument('--valid_thres', type=float, default=0.2) 
    parser.add_argument('--split_type', type=str, default='split', choices=['split', 'continuous'])
    args = parser.parse_args()

    city_name = 'ITRI'
    raw_root = os.path.join(args.root, 'raw_data')
    tracking_root = os.path.join(raw_root, 'tracking')
    seq_id_dic = {}
    traj_len = args.num_historical_steps + args.num_future_steps
    for tracking_file in sorted(os.listdir(tracking_root)):
        print("file_name:", tracking_file)
        with open(os.path.join(tracking_root, tracking_file), 'r') as f:
            data = json.load(f)

        # initialize some variables
        colname = ['TIMESTAMP','TRACK_ID','OBJECT_TYPE','X','Y','CITY_NAME']
        av_df = pd.DataFrame(columns = colname)
        others_df = pd.DataFrame(columns = colname)
        
        objects_id_to_frame_dict = dict() # "id":[[start index1, end index1],[start index2, end index2],...]
        timestamps = list(data['frames'].keys())
        for index, timestamp in enumerate(timestamps):
            objects = data['frames'][timestamp]['objects']
            if len(objects) == 0:
                continue

            av_object = data['frames'][timestamp]['pose']
            row_dic = dict()
            row_dic['TIMESTAMP'] = timestamp
            row_dic['TRACK_ID'] = 'AV'
            row_dic['OBJECT_TYPE'] = 'AV'
            row_dic['X'] = av_object['position']['x']
            row_dic['Y'] = av_object['position']['y']
            row_dic['CITY_NAME'] = city_name
            row_dic = pd.DataFrame.from_dict(row_dic, orient='index').T
            av_df = pd.concat([av_df, row_dic], ignore_index = True)

            ## seem all object_type is others
            for object in objects:
                row_dic = dict()
                row_dic['TIMESTAMP'] = object['timestamp']
                row_dic['TRACK_ID'] = object['tracking_id']
                # row_dic['OBJECT_TYPE'] = 'OTHERS'
                row_dic['OBJECT_TYPE'] = object['tracking_name']
                row_dic['X'] = object['translation']['x']
                row_dic['Y'] = object['translation']['y']
                row_dic['CITY_NAME'] = city_name
                row_dic = pd.DataFrame.from_dict(row_dic, orient='index').T
                others_df = pd.concat([others_df, row_dic], ignore_index = True)

        unique_ts = sorted(others_df['TIMESTAMP'].unique())
        # timestamp ➜ frame index 
        ts_to_index = {ts: idx for idx, ts in enumerate(unique_ts)}
        av_df['TIMESTAMP'] = av_df['TIMESTAMP'].map(ts_to_index)
        others_df['TIMESTAMP'] = others_df['TIMESTAMP'].map(ts_to_index)
        
        unique_obj_type = sorted(others_df['OBJECT_TYPE'].unique())
        for obj_type in unique_obj_type:
            if obj_type not in seq_id_dic:
                seq_id_dic[obj_type] = 0
        
            save_root = os.path.join(args.root, f'av1_dataset_{args.split_type}', obj_type, 'all/data')
            os.makedirs(save_root, exist_ok=True)
            seq_id = seq_id_dic[obj_type]
            seq_id_dic[obj_type] = split_by_ts(av_df, others_df, unique_ts, seq_id, obj_type, \
                                               save_root, args.split_type, args.valid_thres, max_len=50)
