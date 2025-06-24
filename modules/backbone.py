import torch
import torch.nn as nn
import copy
from layers import TwoLayerMLP
from utils import init_weights, compute_angles_lengths_2D, \
    transform_point_to_local_coordinate, transform_point_to_global_coordinate

class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.num_modes = args.num_modes
        self.num_historical_steps = args.num_historical_steps
        self.num_future_steps = args.num_future_steps   
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        hidden_dim = self.hidden_dim

        self.recurr_num = args.recurr_num
        self.recurr_steps = self.num_future_steps // self.recurr_num
        self.key_idxes = [(i*self.recurr_steps)-1 for i in range(1, self.recurr_num+1)]
        
        attn = nn.MultiheadAttention(hidden_dim, self.num_heads, self.dropout, batch_first=True)

        self.proposal2anchor_MLP = TwoLayerMLP(input_dim=self.num_future_steps*2, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)

        delta_feature_in = 6 # 6
        self.delta_MLP = TwoLayerMLP(input_dim=self.recurr_steps*delta_feature_in, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.delta_attn = copy.deepcopy(attn)

        self.m2m_attn = copy.deepcopy(attn)
        self.delta_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.recurr_steps*2)
        
        self.prob_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.prob_norm = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, p_traj, hist_agent, hist_heading):
        '''
        use detail map
        Args:
            p_traj: [B, M, K, F, 2]
            hist_agent: [B, H, 2]
            hist_heading: [B, H]
        '''
        B = p_traj.shape[0]
        M = p_traj.shape[1]
        F_T = self.num_future_steps
        K = self.num_modes
        device = p_traj.device

        init_traj = p_traj.clone() # [B, M, K, F, 2]
        origin = hist_agent[:,-1].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1,M, K,F_T,-1) # [B, M, K, F, 2]
        heading = hist_heading[:,-1].reshape(B, 1, 1, 1).expand(-1,M, K,F_T) # [B, M, K, F]

        traj = init_traj # [B, M, K, F, 2]
        traj = transform_point_to_local_coordinate(traj.reshape(-1,2), \
                            origin.reshape(-1,2), heading.reshape(-1)) # [B*M*K*F, 2]
        traj = traj.reshape(B, M*K, F_T, 2)

        vec = traj[:,:,1:] - traj[:,:,:-1] # [B, M*K, F-1, 2]
        vec = torch.cat([traj[:,:,:1], vec], dim=-2) # [B, M*K, F, 2]
        length, theta = compute_angles_lengths_2D(vec) # [B, M*K, F, 2]

        traj_delta = traj.unsqueeze(2) - traj.unsqueeze(1) # [B, M*K, M*K, F, 2]
        vec_delta = vec.unsqueeze(2) - vec.unsqueeze(1) # [B, M*K, M*K, F, 2]
        length_delta = length.unsqueeze(2) - length.unsqueeze(1) # [B, M*K, M*K, F]
        theta_delta = theta.unsqueeze(2) - theta.unsqueeze(1) # [B, M*K, M*K, F]
        delta_feature = torch.cat([traj_delta, vec_delta, length_delta.unsqueeze(-1), theta_delta.unsqueeze(-1)], dim=-1) # [B, M*K, M*K, F, 6]

        anchor = self.proposal2anchor_MLP(traj.reshape(B, M*K, F_T*2))
        anchor = anchor.reshape(-1, 1, self.hidden_dim) # [B*M*K, 1, D]
        last_idx = 0
        R_T = self.recurr_steps
        m_query = anchor
        refine_traj = torch.tensor([]).to(device)
        for i, idx in enumerate(self.key_idxes):
            delta_query = m_query  # [B*M*K, 1, D]
            delta_feature_i = delta_feature[:,:,:,last_idx:idx+1] # [B, M*K, M*K, R_T, 6]
            delta_feature_i = delta_feature_i.reshape(-1, M*K, R_T*delta_feature_i.shape[-1]) # [B*M*K, M*K, R_T*6]
            delta_embed = self.delta_MLP(delta_feature_i) # [B*M*K, M*K, D]
            m = self.delta_attn(query=delta_query, key=delta_embed, value=delta_embed)[0] # [B*M*K, 1, D]
            
            m = m.reshape(B*M, K, self.hidden_dim)
            m = self.m2m_attn(query=m, key=m, value=m)[0]

            refine_delta_i = self.delta_decoder(m)
            refine_delta_i = transform_point_to_global_coordinate(refine_delta_i.reshape(-1,2), \
                                torch.zeros_like(refine_delta_i).reshape(-1,2), heading[:,:,:,:self.recurr_steps].reshape(-1))
            refine_traj_i = init_traj[:,:,:,last_idx:idx+1] + refine_delta_i.reshape(B, M, self.num_modes, self.recurr_steps, 2)
            
            refine_traj = torch.cat([refine_traj, refine_traj_i], dim=-2) # [B, M, K, F, 2]
            last_idx = idx+1
            m_query = m.reshape(-1, 1, self.hidden_dim)

        refine_pi = self.prob_decoder(m).squeeze(-1)
        refine_pi = self.prob_norm(refine_pi)
        refine_pi = refine_pi.reshape(B, M, K)

        return refine_traj, refine_pi
