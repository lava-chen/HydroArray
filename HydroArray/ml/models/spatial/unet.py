"""
U-Net 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样层，用于U-Net的解码路径
    
    结构:
        1. 上采样: 使用转置卷积将特征图大小翻倍
        2. 特征拼接: 将上采样后的特征图与对应编码路径的特征图拼接
        3. 双卷积: 对拼接后的特征图进行两次卷积操作
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


"""
U-Net 模型

结构:
    编码器: 下采样路径，提取特征
    解码器: 上采样路径，恢复空间分辨率
    跳跃连接: 连接编码器和解码器的特征图
"""
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        for feature in reversed(features):
            self.ups.append(Up(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            (B, out_channels, H, W) 输出图像
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x, skip_connections[idx])

        return self.final_conv(x)


"""
UNet2D 模型实现

结构:
    编码器: 4个Down层，每个Down层包含2个卷积层和1个最大池化层
    解码器: 4个Up层，每个Up层包含1个上采样层和2个卷积层
    瓶颈: 1个DoubleConv层
    输出层: 1个1x1卷积层

参数:
    in_channels: 输入通道数
    out_channels: 输出通道数
    base_filters: 基础通道数，每个Down层的输出通道数为base_filters, base_filters*2, base_filters*4, base_filters*8
"""
class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=64):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, [base_filters, base_filters*2, base_filters*4, base_filters*8])

    def forward(self, x):
        return self.unet(x)