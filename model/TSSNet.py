import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from argoverse.evaluation.competition_util import generate_forecasting_h5

from losses import CELoss
from metrics import BrierMinFDE, MR, MinADE, MinFDE

from modules.backbone import Backbone
from utils.visual_utils import draw_quartiles, AV1Visualize3D
from utils.algo import moving_average_smoothing

class TSSNet(pl.LightningModule):
    def __init__(self, 
                 args,
                 **kwargs
                 ):
        super(TSSNet, self).__init__()
        self.save_hyperparameters()

        self.model = Backbone(args)

        self.num_modes = args.num_modes
        self.num_historical_steps = args.num_historical_steps
        self.num_future_steps = args.num_future_steps   
        self.lr = args.lr
        self.T_max = args.T_max
        self.max_epochs = args.max_epochs
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.iter_num = args.iter_num
        self.best_model_idx = 0 # 0

        self.prob_loss = CELoss()
        self.reg_loss = nn.HuberLoss(delta=0.6, reduction='none')

        self.train_brier_minFDE = BrierMinFDE()
        self.train_minADE = MinADE()
        self.train_minFDE = MinFDE()
        self.train_MR = MR()
        self.val_brier_minFDE = BrierMinFDE()
        self.val_minADE = MinADE()
        self.val_minFDE = MinFDE()
        self.val_MR = MR()

        self.train_loss = []
        self.train_cls_loss = []
        self.val_loss = []
        self.val_cls_loss = []
        self.test_traj_output = dict()
        self.test_prob_output = dict()

        self.split = args.split if hasattr(args, 'split') else None
        self.seed = args.seed
        self.instability_mask = torch.tensor([]).cuda()
        self.total_metric_results = torch.tensor([]).cuda()

    def save_ckpt(self, dir_path, save_input, epoch, split='train'):  
        loss, cls_loss, minade, minfde, mr = save_input
        
        save_txt_dir = dir_path+"/info_v0"
        os.makedirs(save_txt_dir, exist_ok=True)
        save_txt_path = save_txt_dir + "/info_" + str(epoch) + ".txt"

        lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else 0
        tmp_epoch = str(epoch) + '/' + str(self.max_epochs)
        info = {
            'epoch': tmp_epoch,
            'lr': lr,
            '{}_loss'.format(split): loss,
            '{}_cls_loss'.format(split): cls_loss,
            '{}_minADE'.format(split): minade,
            '{}_minFDE'.format(split): minfde,
            '{}_MR'.format(split): mr,
        }

        with open(save_txt_path, 'a') as f:
            for key in info.keys():
                f.write(key+':'+str(info[key])+'\n')
        
        print("save {}_info to {}".format(split, save_txt_path))
        self.epoch_end_cleanup(split=split)

    def epoch_end_cleanup(self, split='train'):
        if split == 'train':
            self.train_loss = []
            self.train_cls_loss = []
            self.train_brier_minFDE.reset()
            self.train_minADE.reset()
            self.train_minFDE.reset()
            self.train_MR.reset()
        elif split == 'val':
            self.val_loss = []
            self.val_cls_loss = []
            self.val_brier_minFDE.reset()
            self.val_minADE.reset()
            self.val_minFDE.reset()
            self.val_MR.reset()

    def calculate_metrics(self, refine_loc, pi, gt_eval, split='train'):
        B = refine_loc.shape[0]
        fde = torch.norm(refine_loc[:, :, -1,:2] - gt_eval[:, -1].unsqueeze(1), p=2, dim=-1)
        best_mode = fde.argmin(dim=-1)
        refine_pred_best = refine_loc[torch.arange(refine_loc.size(0)), best_mode]
        pi_best = pi[torch.arange(B), best_mode] 

        if split == 'train':
            self.train_brier_minFDE.update(refine_pred_best[..., :2], gt_eval, pi_best)
            self.train_minADE.update(refine_pred_best[..., :2], gt_eval)
            self.train_minFDE.update(refine_pred_best[..., :2], gt_eval)
            self.train_MR.update(refine_pred_best[..., :2], gt_eval)
            self.log('train_minADE', self.train_minADE.compute().item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=gt_eval.size(0))
            self.log('train_minFDE', self.train_minFDE.compute().item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=gt_eval.size(0))
            self.log('train_MR', self.train_MR.compute().item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=gt_eval.size(0))
        elif split == 'val':
            self.val_brier_minFDE.update(refine_pred_best[..., :2], gt_eval, pi_best)
            self.val_minADE.update(refine_pred_best[..., :2], gt_eval)
            self.val_minFDE.update(refine_pred_best[..., :2], gt_eval)
            self.val_MR.update(refine_pred_best[..., :2], gt_eval)
            self.log('val_minADE', self.val_minADE.compute().item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.val_minFDE.compute().item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_MR', self.val_MR.compute().item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
    
    def calculate_loss(self, data, agent_index, refine_traj_ls, pi_ls):
        best_model_idx = self.best_model_idx
        target_traj = data['agent']['position'][:, self.num_historical_steps:self.num_historical_steps+self.num_future_steps]
        gt = target_traj[agent_index] # [B, F, 2]

        reg_loss_refine = 0
        cls_loss = 0
        for i in range(self.iter_num):
            refine_traj = refine_traj_ls[i] # [B, M, K, F, 2]
            pi = pi_ls[i]

            M = refine_traj.shape[1]
            gt_exp = gt.unsqueeze(1).expand(-1, M, -1, -1) # [B, M, F, 2]
            error = (torch.norm(refine_traj - gt_exp.unsqueeze(2), p=2, dim=-1)) # [B, M, K, F]
            best_mode = error.mean(dim=-1).argmin(dim=-1) # [B, M]
            refine_best = refine_traj[torch.arange(refine_traj.size(0))[:, None], \
                                        torch.arange(refine_traj.size(1))[None, :], \
                                        best_mode] # [B, M, F, 2]
            reg_loss_refine_i = self.reg_loss(refine_best, gt_exp) # [B, M, F, 2]
            if i == (self.iter_num-1):
                reg_loss_refine_i = reg_loss_refine_i[:,best_model_idx].sum(dim=-1).mean() 
            else:
                reg_loss_refine_i = reg_loss_refine_i.sum(dim=-1).mean()

            prob = pi
            label = best_mode
        
            if i == (self.iter_num-1):
                cls_loss_i = self.prob_loss(prob[:,best_model_idx].reshape(-1, self.num_modes), label[:,best_model_idx].reshape(-1)) 
            else:
                cls_loss_i = self.prob_loss(prob.reshape(-1, self.num_modes), label.reshape(-1))

            reg_loss_refine += reg_loss_refine_i
            cls_loss += cls_loss_i

        reg_loss_refine = reg_loss_refine / self.iter_num
        cls_loss = cls_loss / self.iter_num
        return reg_loss_refine, cls_loss, gt

    def collect_quartiles(self, data, agent_index, refine_traj, init_agent, metric='ADE'):
        target_traj = data['agent']['position'][:, 20:50]
        gt = target_traj[agent_index] # [B, F, 2]
        total_traj = torch.cat([refine_traj[:,:1], init_agent], dim=1)
        if metric=='ADE':
            metric_results = torch.norm(total_traj - gt.unsqueeze(1).unsqueeze(1), p=2, dim=-1).mean(dim=-1) # [B, M+1, K]
        elif metric=='FDE':
            metric_results = torch.norm(total_traj[:,:,:,-1] - gt[:,-1].unsqueeze(1).unsqueeze(1), p=2, dim=-1) # [B, M+1, K]
        metric_results = metric_results.permute(0,2,1) # [B, K, M+1]
        metric_results = metric_results.reshape(-1, total_traj.shape[1]) # [B*K, M+1]
        self.total_metric_results = torch.cat([self.total_metric_results, metric_results], dim=0)

    def forward(self, data):
        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        B = agent_index.shape[0]

        hist_agent = data['agent']['position'][agent_index,:self.num_historical_steps].clone() # [B, H, 2]
        hist_heading = data['agent']['heading'][agent_index,:self.num_historical_steps].clone() # [B, H]
        init_agent = data['init_agent'].reshape(B, -1, self.num_modes, self.num_future_steps, 2).clone() # [B, M, K, F, 2]

        refine_traj_ls = []
        pi_ls = []
        for i in range(self.iter_num):
            if i==0:
                proposal_agent = init_agent
            else:
                proposal_agent = refine_traj.detach().clone()

            refine_traj, pi = self.model(proposal_agent, hist_agent, hist_heading) 

            if not self.training:
                smooth_refine_traj = moving_average_smoothing(refine_traj, window_size=3)
                smooth_refine_traj = smooth_refine_traj.reshape(B, -1, self.num_modes, self.num_future_steps, 2)
                refine_traj_ls.append(smooth_refine_traj)
            else:
                refine_traj_ls.append(refine_traj)
            pi_ls.append(pi)

        # self.calculate_metrics_all_models(data, agent_index, refine_traj_ls[-1], init_agent)
        # self.collect_quartiles(data, agent_index, refine_traj_ls[-1], init_agent, metric='FDE')
        # viz_wrapper = AV1Visualize3D()
        # viz_wrapper.visual_map_agents(data, init_agent, refine_traj_ls[-1], agent_index)

        return agent_index, refine_traj_ls, pi_ls

    def training_step(self, data, batch_idx):
        agent_index, refine_traj_ls, pi_ls = self(data)
        best_model_idx = self.best_model_idx
        reg_loss_refine, cls_loss, gt = self.calculate_loss(data, agent_index, refine_traj_ls, pi_ls)
        refine_loc = refine_traj_ls[-1][:,best_model_idx]
        pi = pi_ls[-1][:,best_model_idx]

        loss = cls_loss + reg_loss_refine * 5
        self.train_loss.append(loss.item())
        self.train_cls_loss.append(cls_loss.item())
        mean_cls_loss = sum(self.train_cls_loss) / len(self.train_cls_loss)
        self.log('cls_loss', mean_cls_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=1)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log('loss', mean_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=1)

        self.calculate_metrics(refine_loc, pi, gt, split='train')

        return loss

    def on_train_epoch_end(self):
        if self.trainer.global_rank == 0:
            brier_minFDE = self.train_brier_minFDE.compute().item()
            minade = self.train_minADE.compute().item()
            minfde = self.train_minFDE.compute().item()
            mr = self.train_MR.compute().item()
            data_len = len(self.train_loss)
            mean_loss = sum(self.train_loss) / data_len
            mean_cls_loss = sum(self.train_cls_loss) / data_len
            
            print("train_loss: ", mean_loss)
            print("train_cls_loss: ", mean_cls_loss)
            print("train_brir_minFDE: ", brier_minFDE)
            print("train_minADE: ", minade, "train_minFDE: ", minfde, "train_MR: ", mr)
            save_input = [mean_loss, mean_cls_loss, minade, minfde, mr]
            self.save_ckpt(self.save_path, save_input, self.current_epoch, split='train')

    def validation_step(self, data, batch_idx):
        agent_index, refine_traj_ls, pi_ls = self(data)
        best_model_idx = self.best_model_idx
        reg_loss_refine, cls_loss, gt = self.calculate_loss(data, agent_index, refine_traj_ls, pi_ls)
        refine_loc = refine_traj_ls[-1][:,best_model_idx]
        pi = pi_ls[-1][:,best_model_idx]

        loss = cls_loss + reg_loss_refine * 5
        self.val_loss.append(loss.item())
        self.val_cls_loss.append(cls_loss.item())
        self.calculate_metrics(refine_loc, pi, gt, split='val')
      
    def on_validation_epoch_end(self):
        # draw_quartiles(self.total_metric_results, metric='FDE')
        # assert 0
        if self.trainer.global_rank == 0:
            print("sample counts: ", self.val_minFDE.count.item())

            brier_minFDE = self.val_brier_minFDE.compute().item()
            minade = self.val_minADE.compute().item()
            minfde = self.val_minFDE.compute().item()
            mr = self.val_MR.compute().item()

            data_len = len(self.val_loss)
            mean_loss = sum(self.val_loss) / data_len
            mean_cls_loss = sum(self.val_cls_loss) / data_len
            print("val_loss: ", mean_loss)
            print("val_cls_loss: ", mean_cls_loss)
            print("val_brier_minFDE: ", brier_minFDE)
            print("val_minADE: ", minade, "val_minFDE: ", minfde, "val_MR: ", mr)
            save_input = [mean_loss, mean_cls_loss, minade, minfde, mr]
            self.save_ckpt(self.save_path, save_input, self.current_epoch, split='val')

    def test_step(self,data,batch_idx):
        agent_index, refine_traj_ls, pi_ls = self(data)

        best_model_idx = self.best_model_idx
        refine_loc = refine_traj_ls[-1][:,best_model_idx]
        pi = pi_ls[-1][:,best_model_idx]

        agent_prob = pi**2
        agent_prob = agent_prob / agent_prob.sum(dim=-1, keepdim=True)
        agent_traj = refine_loc # [B,K,F,2]

        B = agent_traj.shape[0]
        for i in range(B):
            id = int(data['scenario_id'][i])
            traj = agent_traj[i].cpu().numpy()
            prob = agent_prob[i].tolist()

            self.test_traj_output[id] = traj
            self.test_prob_output[id] = prob

    def on_test_end(self):
        output_path = self.save_path # './test_output'
        filename = 'submission'
        generate_forecasting_h5(self.test_traj_output, output_path, filename, self.test_prob_output)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
   
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]