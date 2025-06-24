import os
import pickle
import torch
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.map_representation.itri_map_api import ArgoverseMap

def eval_store_models(scenario_id, pred_path, init_agent):
    '''
    init_agent:[B,K,F,2]
    '''
    ## save prediction of different models
    for i in range(len(scenario_id)):
        seq_id = int(scenario_id[i])

        pred_data = dict()
        if isinstance(init_agent, np.ndarray):
            pred_data['init_agent'] = init_agent[i]
        elif isinstance(init_agent, torch.Tensor):
            pred_data['init_agent'] = init_agent[i].detach().cpu().numpy()
        else:
            raise ValueError('init_agent type error')

        with open(os.path.join(pred_path, f'{seq_id}.pkl'), 'wb') as handle:
            pickle.dump(pred_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_global_map(map_path):
    am = ArgoverseMap()
    lane_dict = am.build_centerline_index()

    dic = {"PIT": [], "MIA": []}
    turn_direction_type = ['NONE', 'LEFT', 'RIGHT']

    # for city_name in ["PIT", "MIA"]:
    for city_name in ["ITRI"]:
        centerline_segments = []
        centerline_vectors = []
        centerline_headings = []
        centerline_lengths = []
        centerline_features = []
        map_data = {}
        for i, lane_id in enumerate(lane_dict[city_name].keys()):
            # extract from API
            lane_cl = am.get_lane_segment_centerline(lane_id, city_name)
            centerline = torch.from_numpy(lane_cl[:, :2]).float() # [P, 2]
            centerline_segment = torch.stack([centerline[:-1], centerline[1:]], dim=1) # [P-1, 2, 2]
            centerline_vector = centerline[1:] - centerline[:-1] # [P-1, 2]
            centerline_heading = torch.atan2(centerline_vector[:, 1], centerline_vector[:, 0]) # [P-1]
            centerline_length = torch.norm(centerline_vector, dim=-1) # [P-1]
            is_intersection = torch.tensor(am.lane_is_in_intersection(lane_id, city_name), dtype=torch.int) 
            turn_direction = torch.tensor(turn_direction_type.index(am.get_lane_turn_direction(lane_id, city_name)), dtype=torch.int)
            traffic_control = torch.tensor(am.lane_has_traffic_control_measure(lane_id, city_name), dtype=torch.int)
            centerline_feature = torch.stack([is_intersection, turn_direction, traffic_control], dim=-1) # [3]
            centerline_feature = centerline_feature.unsqueeze(0).expand(centerline_heading.shape[0], -1) # [P-1, 3]

            centerline_segments.append(centerline_segment)
            centerline_vectors.append(centerline_vector)
            centerline_headings.append(centerline_heading)
            centerline_lengths.append(centerline_length)
            centerline_features.append(centerline_feature)

        map_data['centerline_segments'] = torch.cat(centerline_segments, dim=0)
        map_data['centerline_vectors'] = torch.cat(centerline_vectors, dim=0)
        map_data['centerline_headings'] = torch.cat(centerline_headings, dim=0)
        map_data['centerline_lengths'] = torch.cat(centerline_lengths, dim=0)
        map_data['centerline_features'] = torch.cat(centerline_features, dim=0)
        map_data['num_nodes'] = map_data['centerline_segments'].shape[0]
        dic[city_name] = map_data

    torch.save(dic, map_path)
