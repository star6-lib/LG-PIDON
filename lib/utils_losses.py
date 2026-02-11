import torch.nn as nn
import torch

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

# ----------------------------------------------------------------
#       内部 PDE 残值(用于PDE Loss)
# ----------------------------------------------------------------
def plate_stress_loss(u, v, x, y, b_E, b_nu):
    '''
        Calculate the PDE residual (Equilibrium Equation)
        Input:
            b_E, b_nu: (B, 1) 来自 Branch Net
    '''

    # 1. Broadcasting: (B, 1) -> (B, N)
    E = b_E.expand_as(u)
    nu = b_nu.expand_as(u)

    # 2. Strain
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)

    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)

    # 3. Stress (Plane Stress)
    factor = E / (1 - nu ** 2)
    sigma_xx = factor * (eps_xx + nu * eps_yy)
    sigma_yy = factor * (eps_yy + nu * eps_xx)
    sigma_xy = factor * (1 - nu) * eps_xy

    # 4. Equilibrium
    sigma_xx_x = gradients(sigma_xx, x)
    sigma_xy_y = gradients(sigma_xy, y)
    sigma_xy_x = gradients(sigma_xy, x)
    sigma_yy_y = gradients(sigma_yy, y)

    # Body force = 0
    res_x = sigma_xx_x + sigma_xy_y
    res_y = sigma_xy_x + sigma_yy_y

    return res_x, res_y

# ----------------------------------------------------------------
#       边界应力计算(用于free BC Loss)
# ----------------------------------------------------------------
# 适用于左右边界(法向 n=[1,0] 或 [-1,0])， 自由边界条件: sigma_xx = 0, sigma_xy = 0
def bc_edgeX_loss(u, v,x, y, b_E, b_nu):
    # 1. Broadcasting
    E = b_E.expand_as(u)
    nu = b_nu.expand_as(u)

    # 2. Strain
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)  # 虽然边界上 y 导数可能难求，但自动微分支持

    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)

    # 3. Stress (只计算需要的 sigma_xx 和 sigma_xy)
    factor = E / (1 - nu ** 2)
    sigma_xx = factor * (eps_xx + nu * eps_yy)
    sigma_xy = factor * (1 - nu) * eps_xy

    return sigma_xx, sigma_xy


# 适用于上下边界 (法向 n=[0,1] 或 [0,-1]),自由边界条件: sigma_yy = 0, sigma_xy = 0
def bc_edgeY_loss(u, v, x, y, b_E, b_nu):
    E = b_E.expand_as(u)
    nu = b_nu.expand_as(u)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)

    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)

    factor = E / (1 - nu ** 2)
    sigma_yy = factor * (eps_yy + nu * eps_xx)
    sigma_xy = factor * (1 - nu) * eps_xy

    return sigma_yy, sigma_xy
