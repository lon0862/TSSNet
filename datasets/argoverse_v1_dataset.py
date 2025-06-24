import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.map_representation.itri_map_api import ArgoverseMap
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData

from utils import compute_angles_lengths_2D
from utils.save_utils import save_global_map


class ArgoverseV1Dataset_models(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 processed_root: str,
                 init_pred_root: str,
                 num_historical_steps: int = 20,
                 num_future_steps:int = 30,
                 ) -> None:
        self.root = root
        self.processed_root = processed_root
        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = os.listdir(self.raw_dir)
        print("Number of", split, "dataset: ", len(self._raw_file_names))
        self._processed_file_names = [os.path.splitext(name)[0] + '.pt' for name in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps

        self.init_pred_root = os.path.join(init_pred_root, self._directory)
        self.init_pred_file_names = [os.path.splitext(name)[0] + '.pkl' for name in self.raw_file_names]

        self.global_map_root = os.path.join(self.processed_root, 'global_map.pt')
        if not os.path.exists(self.global_map_root):
            save_global_map(self.global_map_root)
            print('Global map saved at', self.global_map_root)
        self.global_map_data = torch.load(self.global_map_root)

        super(ArgoverseV1Dataset_models, self).__init__(root=root)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.processed_root, self._directory, 'TSSNet_processed')
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            df = pd.read_csv(raw_path)
            data = dict()      
            scenario_id = self.get_scenario_id(raw_path)
            city = self.get_city(df)
            data['city'] = city
            data['scenario_id'] = scenario_id
            data.update(self.get_features(df))
            torch.save(data, os.path.join(self.processed_dir, scenario_id +'.pt'))
    
    @staticmethod
    def get_scenario_id(raw_path: str) -> str:
        return os.path.splitext(os.path.basename(raw_path))[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['CITY_NAME'].values[0]

    def get_features(self, df: pd.DataFrame) -> Dict:
        data = {
            'agent': {},
        }
        ## AGENT
        ## filter out actors that are unseen during the historical time steps
        timestep_ids = list(np.sort(df['TIMESTAMP'].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        agent_ids = list(historical_df['TRACK_ID'].unique())
        num_agents = len(agent_ids)
        df = df[df['TRACK_ID'].isin(agent_ids)]
        
        agent_index = agent_ids.index(df[df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0])

        ## initialization
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        length_mask = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        
        for track_id, track_df in df.groupby('TRACK_ID'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['TIMESTAMP']]

            visible_mask[agent_idx, agent_steps] = True

            length_mask[agent_idx, 0] = False
            length_mask[agent_idx, 1:] = ~(visible_mask[agent_idx, 1:self.num_historical_steps] & visible_mask[agent_idx, :self.num_historical_steps-1])

            agent_position[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df['X'].values, track_df['Y'].values], axis=-1)).float()
            motion = torch.cat([agent_position.new_zeros(1,2), agent_position[agent_idx,1:] - agent_position[agent_idx,:-1]], dim=0)
            length, heading = compute_angles_lengths_2D(motion)
            agent_length[agent_idx] = length[:self.num_historical_steps]
            agent_heading[agent_idx] = heading[:self.num_historical_steps]
            agent_length[agent_idx, length_mask[agent_idx]] = 0
            agent_heading[agent_idx, length_mask[agent_idx]] = 0

        data['agent']['num_nodes'] = num_agents
        data['agent']['agent_index'] = agent_index
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading ## vector heading
        data['agent']['length'] = agent_length ## vector length
        
        return data

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx: int) -> HeteroData:     
        data = torch.load(self.processed_paths[idx])
        file_name = self.init_pred_file_names[idx]

        model_names = ['HPNet', 'DGFNet', 'SmartRefine', 'SIMPL'] ## if av1
        # model_names = ['HPNet', 'SmartRefine'] ## if itri

        init_agent = []
        for model_name in model_names:
            path = os.path.join(self.init_pred_root, model_name, file_name)
            with open(path, 'rb') as handle:
                init_data = pickle.load(handle)
                for key in init_data.keys():
                    if isinstance(init_data[key], np.ndarray):
                        init_data[key] = torch.from_numpy(init_data[key]).float()

            init_agent.append(init_data['init_agent'])
                
        init_agent = torch.stack(init_agent, dim=0) # [M, K, F, 2]
        data['init_agent'] = init_agent
        data['global_map'] = self.global_map_data[data['city']]

        return HeteroData(data)