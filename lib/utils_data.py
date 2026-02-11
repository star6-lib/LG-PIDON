import scipy.io as sio
import numpy as np
import torch

# function to generate data loader for 2D plate stress problem 
def generate_plate_stress_data_loader(args, config):

    # load the data
    mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))
    print(f"Loading data from: ./data/{args.data}.mat ...")

    # Branch Net Inputs
    raw_force = mat_contents['input_force_data'][0]  # List of arrays or (N_samples, 101)
    raw_disp = mat_contents['input_disp_data'][0]  # List of arrays or (N_samples, 101)
    raw_mat = mat_contents['input_material']  # (N_samples, 2) [E, nu]
    raw_geo = mat_contents['geo_param_dict'][0]

    # Trunk Net Inputs
    u_list = mat_contents['final_u'][0]  # list of M elements
    v_list = mat_contents['final_u_y'][0]  # list of M elements
    coors_list = mat_contents['coors_dict'][0]  # list of M elements

    # 原始Flag数据
    flag_BC_load = mat_contents['flag_BC_load_dict'][0]
    flag_BCxy = mat_contents['flag_BCxy_dict'][0]
    flag_BCy = mat_contents['flag_BCy_dict'][0]
    flag_hole = mat_contents['flag_hole_dict'][0]

    datasize = len(u_list)
    print(f"Total samples: {datasize}")

    # ================= 2. 预计算最大节点数 (为流水线 B 做准备) =================
    max_pde = 0;
    max_load = 0;
    max_fix = 0;
    max_free = 0;
    max_hole = 0

    # [新增] 统计 PDE 点数为 0 的样本数量
    zero_pde_count = 0
    total_pde_points = 0

    # 打印前 3 个样本的详细点数情况，帮你直观检查
    print("\n--- Sample Inspection (First 3 samples) ---")

    for i in range(datasize):
        # 计算各部分真实节点数
        n_load = np.sum(flag_BC_load[i])
        n_fix = np.sum(flag_BCxy[i])
        n_free = np.sum(flag_BCy[i])
        n_hole = np.sum(flag_hole[i])

        # PDE点 = 总点数 - 所有边界点
        total_nodes = len(u_list[i])
        n_pde = len(u_list[i]) - (n_load + n_fix + n_free + n_hole)

        if n_pde <= 0:
            zero_pde_count += 1

        total_pde_points += n_pde

        # 打印前几个样本的详情
        if i < 3:
            print(
                f"Sample {i}: Total={total_nodes}, Load={n_load}, Fix={n_fix}, Free={n_free}, Hole={n_hole} -> PDE={n_pde}")

        max_pde = max(max_pde, n_pde)
        max_load = max(max_load, n_load)
        max_fix = max(max_fix, n_fix)
        max_free = max(max_free, n_free)
        max_hole = max(max_hole, n_hole)

        # 转换为整数
    max_pde, max_load = int(max_pde), int(max_load)
    max_fix, max_free, max_hole = int(max_fix), int(max_free), int(max_hole)

    print("\n--- Dataset Statistics ---")
    print(f"Max PDE Nodes per sample: {max_pde}")
    print(f"Max Load Nodes: {max_load}")
    print(f"Max Free Nodes: {max_free}")
    print(f"Max Fix Nodes: {max_fix}")
    print(f"Max Hole Nodes: {max_hole}")
    print(f"Total PDE points in dataset: {total_pde_points}")
    print(f"Samples with 0 PDE points: {zero_pde_count} / {datasize}")

    if max_pde == 0:
        print("\n[CRITICAL WARNING] max_pde is 0! Your flags cover all nodes. No internal points left for PDE loss.")
        print("Please check if 'flag_BC_*' and 'flag_hole' in your .mat file overlap or cover the whole domain.")
    else:
        print("\n[INFO] PDE points exist. Proceeding to data processing...")

    # ================= 3. 定义辅助函数 =================
    # 定义一个辅助函数——用于数据提取与补零
    def pad_data(data, indices, max_len, dim=1):
        valid_data = data[indices]  # 提取有效数据
        pad_len = max_len - len(indices)
        if dim == 1:  # 标量 (u, v)
            padding = np.zeros((pad_len, 1))
        else:  # 坐标 (x, y)
            padding = np.zeros((pad_len, 2))
        return np.concatenate((valid_data, padding), 0)

    # 定义一个辅助函数——专门用于生成 Flag (1和0)
    def pad_flag(indices, max_len):
        n_valid = len(indices)
        # 真点设为 1
        valid = np.ones((n_valid, 1))
        # 补零设为 0
        padding = np.zeros((max_len - n_valid, 1))
        return np.concatenate((valid, padding), 0)

    # ================= 4. 开始循环处理数据 =================
    # --- 流水线 A 的容器 (Branch) ---
    b_force_list, b_disp_list = [], []
    b_E_list, b_nu_list, b_geo_list = [], [], []

    # --- 流水线 B 的容器 (Trunk/Loss) ---
    # 这里的 T 代表 Tensor Ready
    uT, vT, coorT, flagT = [], [], [], []

    for i in range(datasize):
        # ---------------- Pipeline A: 处理 Branch 输入 (无 Padding) ----------------
        # 1. Force (归一化)/ 400        Disp (归一化)/ 100
        b_force_list.append(raw_force[i].flatten() / 400.0)
        b_disp_list.append(raw_disp[i].flatten() / 100.0)

        # 3. Material
        b_E_list.append([raw_mat[i][0]])  # Keep dim [1]
        b_nu_list.append([raw_mat[i][1]])
        # 4. Geometry
        b_geo_list.append(raw_geo[i].reshape(-1))  # (12,)

        # ---------------- Pipeline B: Trunk Net / Loss 输入 (变长，需 Padding) ----------------
        # 强制 reshape 确保 MATLAB 列向量格式 (N, 1) / (N, 2)
        curr_u = u_list[i].reshape(-1, 1)
        curr_v = v_list[i].reshape(-1, 1)
        curr_coor = coors_list[i].reshape(-1, 2)

        # 提取索引
        idx_load = np.where(flag_BC_load[i] == 1)[0]
        idx_fix = np.where(flag_BCxy[i] == 1)[0]
        idx_free = np.where(flag_BCy[i] == 1)[0]
        idx_hole = np.where(flag_hole[i] == 1)[0]
        # PDE 索引是剩下的
        all_bc = np.concatenate([idx_load, idx_fix, idx_free, idx_hole])
        idx_pde = np.setdiff1d(np.arange(len(curr_u[i])), all_bc)
        # print(f"样本{i}：PDE点数={len(idx_pde)}, Load点数={len(idx_load)}, Fix点数={len(idx_fix)}")

        # 再次检查当前样本
        if len(idx_pde) == 0 and max_pde > 0:
            # 如果某个样本居然没有PDE点（但别的样本有），这里会补全 padding，全0
            pass



        # 重组并 Padding u
        uT.append(np.concatenate([
            pad_data(curr_u, idx_pde, max_pde),
            pad_data(curr_u, idx_load, max_load),
            pad_data(curr_u, idx_free, max_free),  # 注意顺序: Free(BCy) -> Fix(BCxy)
            pad_data(curr_u, idx_fix, max_fix),
            pad_data(curr_u, idx_hole, max_hole)
        ], 0))
        # uT.append(u_padded)

        # 重组并 Padding v
        vT.append(np.concatenate([
            pad_data(curr_v, idx_pde, max_pde),
            pad_data(curr_v, idx_load, max_load),
            pad_data(curr_v, idx_free, max_free),
            pad_data(curr_v, idx_fix, max_fix),
            pad_data(curr_v, idx_hole, max_hole)
        ], 0))
        # vT.append(v_padded)

        # 重组并 Padding u
        coorT.append(np.concatenate([
            pad_data(curr_coor, idx_pde, max_pde, dim=2),  # [修正] dim=2
            pad_data(curr_coor, idx_load, max_load, dim=2),
            pad_data(curr_coor, idx_free, max_free, dim=2),
            pad_data(curr_coor, idx_fix, max_fix, dim=2),
            pad_data(curr_coor, idx_hole, max_hole, dim=2)
        ], 0))
        # coorT.append(coors_padded)

        # 生成 FlagT (与 uT 的拼接顺序严格一致)
        flagT.append(np.concatenate([
            pad_flag(idx_pde, max_pde),  # PDE Flag
            pad_flag(idx_load, max_load),  # Load Flag
            pad_flag(idx_free, max_free),  # Free Flag
            pad_flag(idx_fix, max_fix),  # Fix Flag
            pad_flag(idx_hole, max_hole)  # Hole Flag
        ], 0))
        # flagT.append(flag_padded)

    # ================= 5. 转换为 Tensor =================

     # 流水线 A: 直接转
    B_force = torch.tensor(np.array(b_force_list), dtype=torch.float32)
    B_disp = torch.tensor(np.array(b_disp_list), dtype=torch.float32)
    B_E = torch.tensor(np.array(b_E_list), dtype=torch.float32)
    B_nu = torch.tensor(np.array(b_nu_list), dtype=torch.float32)
    B_geo = torch.tensor(np.array(b_geo_list), dtype=torch.float32)

    # 流水线 B: 拼接后转 (Batch, Max_Nodes, Dim)
    U_field = torch.from_numpy(np.stack(uT, 0)).float().squeeze(-1)  # (B, N)
    V_field = torch.from_numpy(np.stack(vT, 0)).float().squeeze(-1)  # (B, N)
    Coor_field = torch.from_numpy(np.stack(coorT, 0)).float()  # (B, N, 2)
    Flag_field = torch.from_numpy(np.stack(flagT, 0)).float().squeeze(-1)  # (B, N)

    # 节点计数（用于Loss切片）
    node_counts = {
        'n_pde': max_pde, 'n_load': max_load,
        'n_free': max_free, 'n_fix': max_fix, 'n_hole': max_hole
    }

    print(f"Branch Shapes: Force{B_force.shape}, E{B_E.shape}, Geo{B_geo.shape}")
    print(f"Trunk Shapes: Coor{Coor_field.shape}, Flag{Flag_field.shape}")

    # split the data
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]

    # 封装 Dataset (包含所有 5 个 Branch 输入 + 4 个 Field 输入)
    train_dataset = torch.utils.data.TensorDataset(
        B_force[bar1[0]:bar1[1]], B_disp[bar1[0]:bar1[1]],B_E[bar1[0]:bar1[1]],
        B_nu[bar1[0]:bar1[1]], B_geo[bar1[0]:bar1[1]],Coor_field[bar1[0]:bar1[1]],
        U_field[bar1[0]:bar1[1]], V_field[bar1[0]:bar1[1]], Flag_field[bar1[0]:bar1[1]]
    )
    val_dataset = torch.utils.data.TensorDataset(
        B_force[bar2[0]:bar2[1]], B_disp[bar2[0]:bar2[1]], B_E[bar2[0]:bar2[1]],
        B_nu[bar2[0]:bar2[1]], B_geo[bar2[0]:bar2[1]], Coor_field[bar2[0]:bar2[1]],
        U_field[bar2[0]:bar2[1]], V_field[bar2[0]:bar2[1]], Flag_field[bar2[0]:bar2[1]]
    )
    test_dataset = torch.utils.data.TensorDataset(
        B_force[bar3[0]:bar3[1]], B_disp[bar3[0]:bar3[1]], B_E[bar3[0]:bar3[1]],
        B_nu[bar3[0]:bar3[1]], B_geo[bar3[0]:bar3[1]], Coor_field[bar3[0]:bar3[1]],
        U_field[bar3[0]:bar3[1]], V_field[bar3[0]:bar3[1]], Flag_field[bar3[0]:bar3[1]]
    )

    # DataLoader (Train shuffle=True, Val/Test Batch=1 for precision evaluation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    return train_loader, val_loader, test_loader, node_counts
