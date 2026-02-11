import torch
import torch.nn as nn
import torch.nn.functional as F
import math

''' ------------------------- LG-PIDON -------------------------- '''
''' ------------------------- Geometry Encoder -------------------------- '''

class DG(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network to encode geometry point cloud
        # Input: (x, y) coordinates of the domain shape
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)
        
    def forward(self, shape_coor, shape_flag):
        '''
       shape_coor: (B, M_geo, 2) - 几何点云坐标
        shape_flag: (B, M_geo)    - 掩码

        return u: (B, 1, F)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)    # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)    # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)    # (B, 1, F)

        return Domain_enc

''' ------------------------- LA-PIDON + GANO Hybrid -------------------------- '''
class GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dim = config['model']['fc_dim']

        # define the geometry encoder
        self.DG = DG(config)

        # 定义一个辅助函数来创建MLP
        def make_branch(in_dim):
            layers = [nn.Linear(in_dim, self.dim), nn.Tanh()]
            for _ in range(config['model']['N_layer'] - 1):
                layers.append(nn.Linear(self.dim, self.dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(self.dim, self.dim))
            return nn.Sequential(*layers)

        # Branch 1：Force Curve(101 points)
        self.branch_force = make_branch(101)
        # Branch 2：Disp Curve(101 points)
        self.branch_disp = make_branch(101)
        # Branch 3：Young's Modulus(1 scalar)
        self.branch_E = make_branch(1)
        # Branch 4：Poisson's Ratio(1 scalar)
        self.branch_nu = make_branch(1)
        # Branch 5：Geometric Parameters(12 scalar)
        self.branch_geo = make_branch(12)

        # Coordinate Lifting
        # 将物理坐标（x, y）升维，以便与DG的Embedding融合
        self.xy_lift_u = nn.Linear(2, self.dim)
        self.xy_lift_v = nn.Linear(2, self.dim)

        # Trunk Net
        # Input dim = Lifted_Coord(F) + DG_Embedding(F) = 2*F
        trunk_in_dim = 2 * self.dim

        def make_trunk():
            layers = [nn.Linear(trunk_in_dim, self.dim), nn.Tanh()]
            for _ in range(config['model']['N_layer'] - 1):
                layers.append(nn.Linear(self.dim, self.dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(self.dim, self.dim))
            return nn.Sequential(*layers)

        self.trunk_u = make_trunk()
        self.trunk_v = make_trunk()

        # Output Heads，为了让u,v独立变化，设置两个独立的线性层
        self.head_u = nn.Linear(self.dim, 1, bias = False)
        self.head_v = nn.Linear(self.dim, 1, bias = False)

    def forward(self,b_force, b_disp, b_E, b_nu, b_geo,     # Branch Inputs
                x_coor, y_coor,                             # Trunk Inputs
                shape_coor, shape_flag):                    # DG Inputs

        # --- A. Branch Net Forward (提取工况特征) ---
        z_f = self.branch_force(b_force)  # (B, F)
        z_d = self.branch_disp(b_disp)  # (B, F)
        z_E = self.branch_E(b_E)  # (B, F)
        z_nu = self.branch_nu(b_nu)  # (B, F)
        z_g = self.branch_geo(b_geo)  # (B, F)

        # [Fusion] Element-wise Product (LA-PIDON 核心机制)
        # 将所有物理场的特征融合为一个全局特征向量
        Z_global = z_f * z_d * z_E * z_nu * z_g  # (B, F)

        # 调整维度以匹配Trunk:(B, 1, F)
        Z_global = Z_global.unsqueeze(1)

        # --- B. Trunk Net Forward (提取时空/几何特征) ---
        # 获取 DG Embedding
        dg_embed = self.DG(shape_coor, shape_flag)  # (B, 1, F)

        # 坐标处理  x_coor: (B, N) -> (B, N, 2)
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)  # (B, N, 2)
        M = xy.shape[1]
        dg_expanded = dg_embed.repeat(1, M, 1)  # (B, N, F)

        # ====== 两个拼装，两个路径 ======
        # [Path U]
        xy_lifted_u = self.xy_lift_u(xy)  # U 的坐标特征
        trunk_input_u = torch.cat((xy_lifted_u, dg_expanded), -1)
        T_features_u = torch.tanh(self.trunk_u(trunk_input_u))

        # [Path V]
        xy_lifted_v = self.xy_lift_v(xy)  # V 的坐标特征
        trunk_input_v = torch.cat((xy_lifted_v, dg_expanded), -1)
        T_features_v = torch.tanh(self.trunk_v(trunk_input_v))

        # ----------- DeepONet Interaction -----------
        # 分别融合
        combined_u = Z_global * T_features_u
        combined_v = Z_global * T_features_v

        # 输出
        u = self.head_u(combined_u).squeeze(-1)
        v = self.head_v(combined_v).squeeze(-1)

        return u, v
