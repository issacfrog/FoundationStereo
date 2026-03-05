# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import EdgeNextConvEncoder

# 视差特征头
class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),                # 3*3卷积特征提取
          nn.ReLU(),                                                                # Relu激活函数
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),  # 两层边缘特征提取
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),  
          nn.Conv2d(input_dim, output_dim, 3, padding=1),                           # 3*3卷积特征提取 最后输出1维（求和，这里实际上省去了通道聚合的操作）
        )

    def forward(self, x):
        return self.conv(x)

# 自定义的一个GRU模型
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        # 针对rqz分别设置卷积层
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    # cz cr cq 是残差项
    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h

# 运动编码器
# 对特征和视差进行特征提取与融合
class BasicMotionEncoder(nn.Module):
    def __init__(self, args, ngroup=8):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (ngroup+1)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+256, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        # 处理相关特征
        cor = F.relu(self.convc1(corr))   # 1*1卷积 提取相关特征
        cor = F.relu(self.convc2(cor))    # 3*3卷积 提取相关特征
        # 处理视差特征
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))
        # 合并相关特征和视差特征
        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1) # 返回融合后的特征与视差

# 一些基础工具
def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


# 大小两种卷积核 分别用来观察细节和拓展视野
class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        super(SelectiveConvGRU, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim+hidden_dim, input_dim+hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        x = self.conv0(x)
        hx = torch.cat([x, h], dim=1)
        hx = self.conv1(hx)
        h = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)

        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        # 运动编码器
        self.encoder = BasicMotionEncoder(args, volume_dim)

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, 256)
        # 掩码
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            )

    # 按照分辨率进行GRU的逐层处理
    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1])) # 处理16倍下采样
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1])) # 处理8倍下采样
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0])) # 处理4倍下采样

        delta_disp = self.disp_head(net[0])

        # 得到掩码
        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp
