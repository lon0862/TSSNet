from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule_models
from model import TSSNet

import warnings
warnings.filterwarnings("ignore")
import torch
torch.set_float32_matmul_precision("high")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--T_max', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=64)
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_historical_steps', type=int, default=20)
    parser.add_argument('--num_future_steps', type=int, default=30)
    parser.add_argument('--num_modes', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0) 
    parser.add_argument('--recurr_num', type=int, default=10)
    parser.add_argument('--iter_num', type=int, default=2)

    parser.add_argument('--root', type=str, default='../../sda/self_driving/argoverse_data1')
    parser.add_argument('--processed_root', type=str, default='../av1_processed')
    parser.add_argument('--init_pred_root', type=str, default='./pred_results_models/pkl')
    parser.add_argument('--save_path', type=str, default='save_ckpt/ckpt_1') 
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True) # 1024

    model = TSSNet(args)    

    datamodule = ArgoverseV1DataModule_models(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=3, mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(devices='0,', accelerator='gpu', 
                         callbacks=[model_checkpoint, lr_monitor], 
                         max_epochs=args.max_epochs,
                         default_root_dir=args.save_path, 
                         accumulate_grad_batches=1, 
                         check_val_every_n_epoch=1, # original 1
                         num_sanity_val_steps=0)
    trainer.fit(model, datamodule)
