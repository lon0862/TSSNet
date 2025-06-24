from argparse import ArgumentParser
import pytorch_lightning as pl

from datasets import ArgoverseV1Dataset_models
from torch_geometric.loader import DataLoader
from model import TSSNet

import warnings
warnings.filterwarnings("ignore")
import torch
torch.set_float32_matmul_precision("high")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=16)
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
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--recurr_num', type=int, default=1)
    parser.add_argument('--iter_num', type=int, default=1)

    parser.add_argument('--root', type=str, default='../../sda/self_driving/argoverse_data1')
    parser.add_argument('--processed_root', type=str, default='../av1_processed')
    parser.add_argument('--init_pred_root', type=str, default='./pred_results_models/pkl')
    
    parser.add_argument('--save_path', type=str, default='./test_output') 
    parser.add_argument('--ckpt_path', type=str, default='pretrained_checkpoints/av1_checkpoint.ckpt')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    
    # checkpoint_dir = 'lightning_logs/version_14/checkpoints/'
    # checkpoint_dir = os.path.join(args.ckpt_path, checkpoint_dir)
    # checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
    checkpoint_path = args.ckpt_path
    print("loading checkpoint from: ", checkpoint_path)

    model = TSSNet.load_from_checkpoint(checkpoint_path, args=args, **vars(args))
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_parameters}")

    test_dataset = ArgoverseV1Dataset_models(args.root, args.split, args.processed_root, \
                        args.init_pred_root, args.num_historical_steps, args.num_future_steps)
    
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,num_workers=args.num_workers, pin_memory=args.pin_memory,persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(devices=args.devices, accelerator='gpu', logger=False)
    trainer.test(model, dataloader)
