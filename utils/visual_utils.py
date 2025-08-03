import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from utils import transform_point_to_local_coordinate

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

class AV1Visualize3D:
    def __init__(self):
        self.model_names = ['HPNet', 'DGFNet', 'SmartRefine', 'SIMPL']
        self.select_idx = 0
        self._SIZE = {
            "vehicle": [4.0, 2.0, 1.8],
        }

    def set_axes_equal(self, ax, ax_limit_settings):
        """讓三個軸的比例一致，避免視覺變形"""
        x_min, x_max, y_min, y_max = ax_limit_settings
        x_middle = (x_min + x_max) / 2
        y_middle = (y_min + y_max) / 2

        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([0, 1])

    def create_fig_and_ax(self, scenario_id, init_agent, refine_agent, gt, hist_agent, metric_results):
        init_minFDE, init_minADE, refine_minFDE, refine_minADE = metric_results

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        
        dpi = 100
        fig.set_dpi(dpi)
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.set_tight_layout(True)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_axis_off()

        title = f'scenario_id: {scenario_id}\n'
        title += f'Baseline minADE: {init_minADE.item(): .3f} / minFDE: {init_minFDE.item(): .3f}\n'
        title += f'TSSNet minADE: {refine_minADE.item(): .3f} / minFDE: {refine_minFDE.item(): .3f}\n'
        ax.set_title(title, fontsize=7, color="black")

        np_init = init_agent.detach().cpu().numpy() # [K, F, 2]
        np_refine = refine_agent.detach().cpu().numpy() # [K, F, 2]
        np_gt = gt.detach().cpu().numpy()
        np_hist = hist_agent.detach().cpu().numpy()
        all_x = np.concatenate([np_refine[:,:,0].flatten(), np_init[:,:,0].flatten(), np_gt[:,0], np_hist[:,0]], axis=0)
        all_y = np.concatenate([np_refine[:,:,1].flatten(), np_init[:,:,1].flatten(), np_gt[:,1], np_hist[:,1]], axis=0)
        thres = 3 #3
        x_min, x_max = all_x.min()-thres, all_x.max()+thres
        y_min, y_max = all_y.min()-thres, all_y.max()+thres

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax_limit_settings = [x_min, x_max, y_min, y_max]
        self.set_axes_equal(ax, ax_limit_settings)
        
        return fig, ax, ax_limit_settings
    
    def plot_map(self, center_seg_i, ax, ax_limit_settings):
        x_min, x_max, y_min, y_max = ax_limit_settings
        x_min -= 100
        x_max += 100
        y_min -= 100
        y_max += 100
        for j in range(len(center_seg_i)):
            center_seg_i_j = center_seg_i[j] # [2, 2]
            if center_seg_i_j[0,0] < x_min and center_seg_i_j[1,0] < x_min:
                continue
            if center_seg_i_j[0,0] > x_max and center_seg_i_j[1,0] > x_max:
                continue
            if center_seg_i_j[0,1] < y_min and center_seg_i_j[1,1] < y_min:
                continue
            if center_seg_i_j[0,1] > y_max and center_seg_i_j[1,1] > y_max:
                continue
            ax.plot(
                center_seg_i_j[:, 0], center_seg_i_j[:, 1], 
                ':',
                zs=0, zdir='z',
                color='#0A1931',
                linewidth=0.3,
                alpha=1,
            )

    def plot_vehicle(self, ax, traj, ts, is_agent):
        l = self._SIZE['vehicle'][0]
        w = self._SIZE['vehicle'][1]
        h = self._SIZE['vehicle'][2] / 30 # 30
        x = traj[ts, 0]
        y = traj[ts, 1]

        start_idx = max(0, ts - 10)
        vector = traj[ts] - traj[start_idx]
        theta = np.arctan2(vector[1], vector[0]) * 180 / np.pi


        ## Rotate the vertices
        theta = np.arctan2(vector[1], vector[0])

        local_vertices = np.array([
            [-l/2, -w/2, 0],
            [ l/2, -w/2, 0],
            [ l/2,  w/2, 0],
            [-l/2,  w/2, 0],
            [-l/2, -w/2, h],
            [ l/2, -w/2, h],
            [ l/2,  w/2, h],
            [-l/2,  w/2, h],
        ])  # shape (8, 3)

        ## Rotate the vertices
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        rotated_xy = np.dot(local_vertices[:, :2], rotation_matrix.T)
        rotated_xy += np.array([x, y])
        rotated_vertices = np.hstack([rotated_xy, local_vertices[:, 2:]])

        # Define the faces of the 3D rectangle
        faces = [[rotated_vertices[j] for j in [0, 1, 2, 3]],
                    [rotated_vertices[j] for j in [4, 5, 6, 7]],
                    [rotated_vertices[j] for j in [0, 3, 7, 4]],
                    [rotated_vertices[j] for j in [1, 2, 6, 5]],
                    [rotated_vertices[j] for j in [0, 1, 5, 4]],
                    [rotated_vertices[j] for j in [2, 3, 7, 6]]]

        # Create a Poly3DCollection
        if is_agent: 
            edgecolor = "#A13033" # 'royalblue'
            facecolor = "#ED8199"
        else:
            edgecolor = "#3C736D"
            facecolor = "#94D7D1"

        poly3d = Poly3DCollection(faces, linewidths=0.4, edgecolors=edgecolor, facecolors=facecolor, alpha=0.3)
        ax.add_collection3d(poly3d)

    def plot_prediction(self, ax, pred_traj, model_name):
        '''
        pred_traj: Tensor: [K,F,2]
        '''
        if model_name == 'GT':
            color = 'red'
            pred_traj = pred_traj.unsqueeze(0)
        elif model_name == 'TSSNet':
            color = 'blue'
        else:
            color = 'orange'

        pred_traj = pred_traj.detach().cpu().numpy()
        for i in range(pred_traj.shape[0]):
            ax.plot(
                pred_traj[i, :, 0], pred_traj[i, :, 1], zs=0, zdir='z', 
                color=color,
                linestyle='dashed',
                linewidth=0.5,
                alpha=1,
                zorder=1000
            )
            dx = pred_traj[i, -1, 0] - pred_traj[i, -2, 0]
            dy = pred_traj[i, -1, 1] - pred_traj[i, -2, 1]
            ax.arrow3D(
                pred_traj[i, -1, 0], pred_traj[i, -1, 1], 0,
                dx, dy, 0,
                mutation_scale=5,
                linewidth=0.65,
                fc=color,
                ec=color,
                label=model_name,
                zorder=1000
            )

    def matplot_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7.5,
            loc='upper right',
            facecolor='white',
        )
        legend.set_zorder(999)

    def visual_map_agents(self, data, init_agent, refine_agent, agent_index):
        '''
        init_agent: [B,M,K,F,2]
        refine_agent: [B,M,K,F,2]
        '''
        hist_agent = data['agent']['position'][agent_index, :20] # [B, H, 2]
        gt = data['agent']['position'][agent_index, 20:50] # [B, F, 2]

        B = init_agent.shape[0]
        K = init_agent.shape[2]

        init_agent_selected = init_agent[:, self.select_idx] # [B,K,F,2]
        refine_agent_selected = refine_agent[:, self.select_idx] # [B,K,F,2]
        init_agent = init_agent.reshape(B,-1,30,2) # [B,M*K,F,2]
        refine_agent = refine_agent.reshape(B,-1,30,2) # [B,M*K,F,2]

        # scenario_id_ls = [17859, 4824, 30197, 8423, 26325, 37549, 1123, 22264, \
        #                   35615, 23424, 3254, 34261, 14377]
        # scenario_id_ls = [3254, 30197, 26325, 23424]
        # id_ls_speed_up = [9500, 1552, 31421, 35752, 36035] # speed up
        # id_ls_turn_direction = [490] # turn direction
        # id_ls_speed_up_serious = [1950, 2723, 10381] # scerious speed up
        # scenario_id_ls = id_ls_speed_up + id_ls_turn_direction + id_ls_speed_up_serious
        # scenario_id_ls = [20541, 17859]
        
        scenario_id_ls = [3254]
        for i in range(B):
            scenario_id = data['scenario_id'][i] 
            if int(scenario_id) not in scenario_id_ls:
                continue
            center_seg_i = data[i]['global_map']['centerline_segments'].detach().cpu().numpy() # [N, 2, 2]
            
            hist_traj_i = data[i]['agent']['position'][:, :20].detach().cpu().numpy() # [N,H,2]
            hist_agent_i = hist_agent[i].detach().cpu().numpy() # [H,2]

            init_norm = torch.norm(init_agent_selected[i] - gt[i].unsqueeze(0), dim=-1) # [K,F]
            init_minFDE, init_best_mode = init_norm[:,-1].min(dim=-1)
            init_minADE = init_norm[init_best_mode].mean(dim=-1)

            refine_norm = torch.norm(refine_agent_selected[i] - gt[i].unsqueeze(0), p=2, dim=-1) # [K,F]
            refine_minFDE, refine_best_mode = refine_norm[:,-1].min(dim=-1)
            refine_minADE = refine_norm[refine_best_mode].mean(dim=-1)
            metric_results = init_minFDE, init_minADE, refine_minFDE, refine_minADE 

            # thres = 3
            # mask = init_minFDE - refine_minFDE
            # if mask <= thres:
            #     continue

            ## plot centerline in x-y plane, in range of x_min, x_max, y_min, y_max
            for ts in range(hist_traj_i.shape[1]+3):
                # if ts < hist_traj_i.shape[1]+2:
                if ts != hist_traj_i.shape[1]+1:
                    continue
                # fig, ax, ax_limit_settings = self.create_fig_and_ax(scenario_id, init_agent_selected[i], \
                #                              refine_agent_selected[i], gt[i], hist_agent[i], metric_results)
                fig, ax, ax_limit_settings = self.create_fig_and_ax(scenario_id, init_agent[i], \
                                             refine_agent[i], gt[i], hist_agent[i], metric_results)
                
                if int(scenario_id) == 30197:
                    ax.view_init(elev=34, azim=23)
                elif int(scenario_id) == 26325:
                    ax.view_init(elev=30, azim=-60)
                elif int(scenario_id) == 3254:
                    ax.view_init(elev=66, azim=-53)
                elif int(scenario_id) == 23424:
                    ax.view_init(elev=66, azim=-53)
                else:
                    ax.view_init(elev=70, azim=-51)
                self.plot_map(center_seg_i, ax, ax_limit_settings)
                ## plot hist vehicle
                for traj in hist_traj_i:
                    self.plot_vehicle(ax, traj, min(ts,hist_traj_i.shape[1]-1), is_agent=False)

                self.plot_vehicle(ax, hist_agent_i, min(ts,hist_traj_i.shape[1]-1), is_agent=True)
                ## plot hist_traj of agent
                past_traj = hist_agent_i[:min(ts,hist_traj_i.shape[1]-1)+1, :2]
                ax.plot(
                    past_traj[:, 0], past_traj[:, 1],
                    color='red',
                    linewidth=0.5,
                )
            
                if ts >= hist_traj_i.shape[1]:
                    self.plot_prediction(ax, gt[i], model_name='GT')
                if ts >= (hist_traj_i.shape[1]+1):
                    self.plot_prediction(ax, init_agent_selected[i], model_name=self.model_names[self.select_idx])
                if ts == (hist_traj_i.shape[1]+2):
                    self.plot_prediction(ax, refine_agent_selected[i], model_name='TSSNet')
                    

                self.matplot_legend(ax)
                plt.show()
                # save_path = './visualize_results/with_title'
                # os.makedirs(save_path, exist_ok=True)
                # plt.savefig(os.path.join(save_path, f'{scenario_id}.png'), dpi=800, bbox_inches='tight', pad_inches=0)

                # save_path = f'./visualize_results/{scenario_id}'
                # os.makedirs(save_path, exist_ok=True)
                # plt.savefig(os.path.join(save_path, f'{ts}.png'), dpi=800, bbox_inches='tight', pad_inches=0)
                plt.clf()
                plt.close()
                assert 0

class AV1Visualize3D_local:
    def __init__(self):
        self.model_names = ['HPNet', 'DGFNet', 'SmartRefine', 'SIMPL']
        self.select_idx = 0
        self._SIZE = {
            "vehicle": [4.0, 2.0, 1.8],
        }

    def set_axes_equal(self, ax, ax_limit_settings):
        """讓三個軸的比例一致，避免視覺變形"""
        x_min, x_max, y_min, y_max = ax_limit_settings
        x_middle = (x_min + x_max) / 2
        y_middle = (y_min + y_max) / 2

        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([0, 1])

    def create_fig_and_ax(self, scenario_id, init_agent, gt, hist_agent):
        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        
        dpi = 100
        fig.set_dpi(dpi)
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.set_tight_layout(True)
        ax.grid(False)
        # ax.set_xticks([])
        # ax.set_xticklabels([])
        # ax.set_yticks([])
        # ax.set_yticklabels([])
        ax.set_zticks([])
        ax.set_zticklabels([])
        # ax.set_axis_off()

        title = f'scenario_id: {scenario_id}\n'
        ax.set_title(title, fontsize=7, color="black")

        np_init = init_agent.detach().cpu().numpy() # [K, F, 2]
        np_gt = gt.detach().cpu().numpy()
        np_hist = hist_agent.detach().cpu().numpy()
        all_x = np.concatenate([np_init[:,:,0].flatten(), np_gt[:,0], np_hist[:,0]], axis=0)
        all_y = np.concatenate([np_init[:,:,1].flatten(), np_gt[:,1], np_hist[:,1]], axis=0)
        thres = 3 #3
        x_min, x_max = all_x.min()-thres, all_x.max()+thres
        y_min, y_max = all_y.min()-thres, all_y.max()+thres

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax_limit_settings = [x_min, x_max, y_min, y_max]
        self.set_axes_equal(ax, ax_limit_settings)
        
        return fig, ax, ax_limit_settings
    
    def plot_map(self, center_seg_i, ax, ax_limit_settings):
        x_min, x_max, y_min, y_max = ax_limit_settings
        x_min -= 100
        x_max += 100
        y_min -= 100
        y_max += 100
        for j in range(len(center_seg_i)):
            center_seg_i_j = center_seg_i[j] # [2, 2]
            if center_seg_i_j[0,0] < x_min and center_seg_i_j[1,0] < x_min:
                continue
            if center_seg_i_j[0,0] > x_max and center_seg_i_j[1,0] > x_max:
                continue
            if center_seg_i_j[0,1] < y_min and center_seg_i_j[1,1] < y_min:
                continue
            if center_seg_i_j[0,1] > y_max and center_seg_i_j[1,1] > y_max:
                continue
            ax.plot(
                center_seg_i_j[:, 0], center_seg_i_j[:, 1], 
                ':',
                zs=0, zdir='z',
                color='#0A1931',
                linewidth=0.3,
                alpha=1,
            )

    def plot_vehicle(self, ax, traj, ts, is_agent):
        l = self._SIZE['vehicle'][0]
        w = self._SIZE['vehicle'][1]
        h = self._SIZE['vehicle'][2] / 30 # 30
        x = traj[ts, 0]
        y = traj[ts, 1]

        start_idx = max(0, ts - 10)
        vector = traj[ts] - traj[start_idx]
        theta = np.arctan2(vector[1], vector[0]) * 180 / np.pi


        ## Rotate the vertices
        theta = np.arctan2(vector[1], vector[0])

        local_vertices = np.array([
            [-l/2, -w/2, 0],
            [ l/2, -w/2, 0],
            [ l/2,  w/2, 0],
            [-l/2,  w/2, 0],
            [-l/2, -w/2, h],
            [ l/2, -w/2, h],
            [ l/2,  w/2, h],
            [-l/2,  w/2, h],
        ])  # shape (8, 3)

        ## Rotate the vertices
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        rotated_xy = np.dot(local_vertices[:, :2], rotation_matrix.T)
        rotated_xy += np.array([x, y])
        rotated_vertices = np.hstack([rotated_xy, local_vertices[:, 2:]])

        # Define the faces of the 3D rectangle
        faces = [[rotated_vertices[j] for j in [0, 1, 2, 3]],
                    [rotated_vertices[j] for j in [4, 5, 6, 7]],
                    [rotated_vertices[j] for j in [0, 3, 7, 4]],
                    [rotated_vertices[j] for j in [1, 2, 6, 5]],
                    [rotated_vertices[j] for j in [0, 1, 5, 4]],
                    [rotated_vertices[j] for j in [2, 3, 7, 6]]]

        # Create a Poly3DCollection
        if is_agent: 
            edgecolor = "#A13033" # 'royalblue'
            facecolor = "#ED8199"
        else:
            edgecolor = "#3C736D"
            facecolor = "#94D7D1"

        poly3d = Poly3DCollection(faces, linewidths=0.4, edgecolors=edgecolor, facecolors=facecolor, alpha=0.3)
        ax.add_collection3d(poly3d)

    def plot_prediction(self, ax, pred_traj, model_name):
        '''
        pred_traj: Tensor: [K,F,2]
        '''
        if model_name == 'GT':
            color = 'red'
            pred_traj = pred_traj.unsqueeze(0)
        elif model_name == 'TSSNet':
            color = 'blue'
        else:
            color = 'orange'

        pred_traj = pred_traj.detach().cpu().numpy()
        for i in range(pred_traj.shape[0]):
            ax.plot(
                pred_traj[i, :, 0], pred_traj[i, :, 1], zs=0, zdir='z', 
                color=color,
                linestyle='dashed',
                linewidth=0.5,
                alpha=1,
                zorder=1000
            )
            dx = pred_traj[i, -1, 0] - pred_traj[i, -2, 0]
            dy = pred_traj[i, -1, 1] - pred_traj[i, -2, 1]
            ax.arrow3D(
                pred_traj[i, -1, 0], pred_traj[i, -1, 1], 0,
                dx, dy, 0,
                mutation_scale=5,
                linewidth=0.65,
                fc=color,
                ec=color,
                label=model_name,
                zorder=1000
            )

    def matplot_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7.5,
            loc='upper right',
            facecolor='white',
        )
        legend.set_zorder(999)

    def transform_traj_to_local(self, traj, origin, heading, B):
        if B == 1:
            traj = traj.unsqueeze(0)

        traj_shape = traj.shape
        traj = traj.reshape(traj_shape[0], -1, traj_shape[-1]) # [B, L, 2]
        origin_exp = origin.unsqueeze(1).expand(-1, traj.shape[1], -1)
        heading_exp = heading.unsqueeze(1).expand(-1, traj.shape[1])
        traj = transform_point_to_local_coordinate(traj.reshape(-1,2), \
                            origin_exp.reshape(-1,2), heading_exp.reshape(-1)) # [B*L, 2]
        traj = traj.reshape(*traj_shape)

        if B == 1:
            traj = traj.squeeze(0)

        return traj

    def visual_map_agents(self, data, init_agent, refine_agent, agent_index):
        '''
        init_agent: [B,M,K,F,2]
        refine_agent: [B,M,K,F,2]
        '''
        hist_agent = data['agent']['position'][agent_index, :20] # [B, H, 2]
        gt = data['agent']['position'][agent_index, 20:50] # [B, F, 2]
        init_agent_selected = init_agent[:, self.select_idx] # [B,K,F,2]

        origin = hist_agent[:,-1] # [B, 2]
        hist_heading = data['agent']['heading'][agent_index,:20].clone() # [B, H]
        heading = hist_heading[:,-1] # [B]

        ## transform global to local coordinate
        hist_agent = self.transform_traj_to_local(hist_agent, origin, heading, hist_agent.shape[0])
        gt = self.transform_traj_to_local(gt, origin, heading, gt.shape[0])
        init_agent_selected = self.transform_traj_to_local(init_agent_selected, origin, heading, init_agent_selected.shape[0])

        scenario_id_ls = [3254]
        for i in range(init_agent.shape[0]):
            scenario_id = data['scenario_id'][i] 
            # if int(scenario_id) not in scenario_id_ls:
            #     continue
            center_seg_i = data[i]['global_map']['centerline_segments'] # [N, 2, 2]
            hist_traj_i = data[i]['agent']['position'][:, :20] # [N,H,2]

            ## transform global to local coordinate
            center_seg_i = self.transform_traj_to_local(center_seg_i, origin[i:i+1], heading[i:i+1], 1)
            hist_traj_i = self.transform_traj_to_local(hist_traj_i, origin[i:i+1], heading[i:i+1], 1)

            center_seg_i = center_seg_i.detach().cpu().numpy()
            hist_traj_i = hist_traj_i.detach().cpu().numpy()
            hist_agent_i = hist_agent[i].detach().cpu().numpy() # [H,2]

            ## plot centerline in x-y plane, in range of x_min, x_max, y_min, y_max
            for ts in range(hist_traj_i.shape[1]+2):
                # if ts < hist_traj_i.shape[1]+2:
                if ts != hist_traj_i.shape[1]+1:
                    continue
                fig, ax, ax_limit_settings = self.create_fig_and_ax(scenario_id, init_agent_selected[i], \
                                             gt[i], hist_agent[i])
                
                ax.view_init(elev=90, azim=-90)
                self.plot_map(center_seg_i, ax, ax_limit_settings)
                ## plot hist vehicle
                for traj in hist_traj_i:
                    self.plot_vehicle(ax, traj, min(ts,hist_traj_i.shape[1]-1), is_agent=False)

                self.plot_vehicle(ax, hist_agent_i, min(ts,hist_traj_i.shape[1]-1), is_agent=True)
                ## plot hist_traj of agent
                past_traj = hist_agent_i[:min(ts,hist_traj_i.shape[1]-1)+1, :2]
                ax.plot(
                    past_traj[:, 0], past_traj[:, 1],
                    color='red',
                    linewidth=0.5,
                )
            
                if ts >= hist_traj_i.shape[1]:
                    self.plot_prediction(ax, gt[i], model_name='GT')
                if ts >= (hist_traj_i.shape[1]+1):
                    self.plot_prediction(ax, init_agent_selected[i], model_name=self.model_names[self.select_idx])

                self.matplot_legend(ax)
                plt.show()
                # assert 0

def draw_quartiles(total_ADE, metric='ADE'):
    q1 = torch.quantile(total_ADE, 0.25, dim=0)
    q2 = torch.quantile(total_ADE, 0.5, dim=0)
    q3 = torch.quantile(total_ADE, 0.75, dim=0)
    print("Q1:", q1)
    print("Q2:", q2)
    print("Q3:", q3)

    model_names = ['TSSNet', 'HPNet', 'DGFNet', 'SmartRefine', 'SIMPL']
    model_num = total_ADE.shape[1]
    data_list = [total_ADE[:, i].tolist() for i in range(model_num)]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, 0, '', labels=[f'{model_names[i]}' for i in range(model_num)], widths=0.5)

    # 加上 Q1/Q2/Q3 標註
    x_offset = 0
    y_offset = 0.25 # 0.15
    for i in range(model_num):
        x = i + 1  # x 軸位置（boxplot 預設從 1 開始編號）
        plt.text(x + x_offset, q1[i] + y_offset, f'Q1: {q1[i]:.2f}', va='center', ha='center', color='blue', fontsize='x-large')
        plt.text(x + x_offset, q2[i] + y_offset, f'Q2: {q2[i]:.2f}', va='center', ha='center', color='blue', fontsize='x-large')
        plt.text(x + x_offset, q3[i] + y_offset, f'Q3: {q3[i]:.2f}', va='center', ha='center', color='blue', fontsize='x-large')

    # if metric=='ADE':
        # plt.ylim(0, 1.4)
    # plt.title(f'Quartiles for {metric}',fontsize='xx-large')
    # plt.xlabel('Model_Name', fontsize='xx-large')
    # plt.ylabel('Value', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./img/quartiles_{metric}.png', dpi=800)
    plt.clf()
    plt.close()

def draw_curve(results, metric='minFDE'):
    len_step = 2.5
    max_len = 26
    X_list = [i*len_step/3 for i in range(max_len)]
    Y_list = []
    for i in range(len(results)):
        minFDE = results[i].mean().item()
        Y_list.append(minFDE)

    ## draw curve
    plt.figure(figsize=(12, 6))
    plt.plot(X_list, Y_list, color='black')
    if metric == 'minFDE':
        plt.axhline(y=0.824, color='red', linestyle='--')
        plt.ylabel("avg minFDE(m)")
    elif metric == 'minADE':
        plt.axhline(y=0.611, color='red', linestyle='--')
        plt.ylabel("avg minADE(m)")

    plt.xlabel("velocity(m/s)")
    plt.show()

def draw_speed_scatter(results, gt_speed):
    results = results.detach().cpu().numpy()
    gt_speed = gt_speed.detach().cpu().numpy()

    data = pd.DataFrame({
        'results': results,
        'gt_speed': gt_speed
    })

    # Step 1: 計算 X 軸 (gt_speed) 的 mean 和 std
    mean_speed = data['gt_speed'].mean()
    std_speed = data['gt_speed'].std()
    x_lower = mean_speed - 5 * std_speed
    x_upper = mean_speed + 5 * std_speed

    # Step 2: 計算 Y 軸 (minFDE or minADE) 的 mean 和 std
    mean_fde = data['results'].mean()
    std_fde = data['results'].std()
    y_lower = mean_fde - 5 * std_fde
    y_upper = mean_fde + 5 * std_fde

    # Step 3: 同時篩選兩個條件
    filtered_data = data[
        (data['gt_speed'] >= x_lower) & (data['gt_speed'] <= x_upper) &
        (data['results'] >= y_lower) & (data['results'] <= y_upper)
    ]

    print("number of data before filter:", len(data['results']))
    print("number of data after filter:", len(filtered_data['results']))

    plt.figure()
    # plt.scatter(gt_speed, results, color='blue', alpha=0.05)
    plt.scatter(filtered_data['gt_speed'], filtered_data['results'], color='blue', alpha=0.05)
    plt.axhline(y=0.824, color='red', linestyle='--')

    plt.xlabel("speed(m/s)")
    plt.ylabel("avg minFDE(m)")
    plt.show()