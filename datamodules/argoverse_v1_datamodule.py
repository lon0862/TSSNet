from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV1Dataset_models

class ArgoverseV1DataModule_models(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 processed_root: str,
                 init_pred_root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30,
                 **kwargs) -> None:
        super(ArgoverseV1DataModule_models, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.processed_root = processed_root
        self.init_pred_root = init_pred_root
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def prepare_data(self) -> None:
        ArgoverseV1Dataset_models(self.root, 'train', self.processed_root, self.init_pred_root, self.num_historical_steps, self.num_future_steps)
        ArgoverseV1Dataset_models(self.root, 'val', self.processed_root, self.init_pred_root, self.num_historical_steps, self.num_future_steps)
        # ArgoverseV1Dataset_models(self.root, 'test', self.processed_root, self.init_pred_root, self.num_historical_steps, self.num_future_steps)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV1Dataset_models(self.root, 'train', self.processed_root, self.init_pred_root, self.num_historical_steps, self.num_future_steps)
        self.val_dataset = ArgoverseV1Dataset_models(self.root, 'val', self.processed_root, self.init_pred_root, self.num_historical_steps, self.num_future_steps)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)