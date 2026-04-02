"""
Diffusion Model 实现

"""

import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    """
    扩散模型
    
    结构:
        编码器: 从输入到潜在空间
        扩散过程: 噪声添加
        解码器: 从潜在空间到输出
    """
    def __init__(self, in_channels=1, time_steps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels

        self.beta = torch.linspace(beta_start, beta_end, time_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        raise NotImplementedError("Subclass must implement forward method")

    def diffusion_step(self, x_t, t):
        noise = torch.randn_like(x_t)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_bar_t)
        return x_0, noise