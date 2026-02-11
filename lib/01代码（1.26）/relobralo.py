import torch
import torch.nn.functional as F
import numpy as np


class ReLoBRaLo:
    def __init__(self, num_losses, alpha=0.999, temperature=0.1, rho=0.99, device='cpu'):
        """
        Args:
            num_losses (int): 损失项的数量 (在你的项目中是4: pde, load, fix, free)
            alpha (float): 指数衰减率 (0.9 - 0.999)
            temperature (float): Softmax温度系数 (T)
            rho (float): 伯努利概率，用于随机回溯 (Bernoulli probability)
            device: torch device
        """
        self.num_losses = num_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.device = device

        # 初始化权重 (归一化为和为 num_losses)
        self.lambdas = torch.ones(num_losses).to(device)

        # 历史记录
        self.last_losses = None  # L(t-1)
        self.init_losses = None  # L(0)

    def update(self, current_losses):
        """
        Args:
            current_losses (list or tensor): 当前step的各个损失项的值 (detach之后的值)
        Returns:
            new_weights (tensor): 用于加权求和的权重
        """
        # 将输入转换为tensor并确保不需要梯度
        if isinstance(current_losses, list):
            current_losses = torch.tensor(current_losses).to(self.device)

        # 第一次迭代，只记录初始值，不更新权重
        if self.init_losses is None:
            self.init_losses = current_losses
            self.last_losses = current_losses
            return self.lambdas

        # 1. 计算相对平衡 (Relative Balance)
        # 避免除以0，加一个小epsilon
        eps = 1e-8

        # L(t) / L(t-1)
        short_term_ratio = current_losses / (self.last_losses + eps)
        # L(t) / L(0)
        long_term_ratio = current_losses / (self.init_losses + eps)

        # Softmax Normalization
        # lambda_bal(t, t-1)
        l_bal_short = self.num_losses * F.softmax(short_term_ratio / self.temperature, dim=0)
        # lambda_bal(t, 0)
        l_bal_long = self.num_losses * F.softmax(long_term_ratio / self.temperature, dim=0)

        # 2. 随机回溯 (Random Lookback / Saudade)
        # 采样 rho: 1 (看上一步) 或 0 (看初始)
        # 注意：原文公式是 rho * lambda(t-1) + (1-rho) * lambda_bal(t,0)
        # 这里 rho 是概率，我们需要生成一个 0/1 变量
        do_lookback = (np.random.rand() > self.rho)  # 如果随机数大于rho，则回溯(使用long term)

        if not do_lookback:
            # 正常路径 (History is t-1)
            lambda_hist = self.lambdas
        else:
            # 回溯路径 (History is reset based on long term balance)
            lambda_hist = l_bal_long

        # 3. 指数衰减更新 (Exponential Decay)
        # lambda(t) = alpha * lambda_hist + (1-alpha) * lambda_bal_short
        new_lambdas = self.alpha * lambda_hist + (1 - self.alpha) * l_bal_short

        # 更新历史状态
        self.last_losses = current_losses
        self.lambdas = new_lambdas.detach()  # 权重本身不参与梯度计算

        return self.lambdas