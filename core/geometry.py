# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        # 金字塔层数
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d).contiguous() #执行geo重排

        # 金字塔的初始化
        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2]) #avg_pool2d 2*2区域求平均pooling
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    # 根据当前的视差估计（Disparity），从不同尺度的代价体（Cost Volume）中提取局部特征，
    # 并将其拼接成一个丰富的特征向量，供后续的优化模块（如 GRU）使用。
    def __call__(self, disp, coords, low_memory=False):
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device) # 将数据移到对应设备上
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            # 传入进来的是dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
            # 基于 corr_radius 生成离散位移向量，用于几何编码体采样。 本质是传入的搜索空间
            # 通过加法实际上是指定了中心位置，和搜索范围
            x0 = self.dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i # 通过2^i控制分辨率，分辨率实际上有粗到精
            y0 = torch.zeros_like(x0)

            # 拼接成[x0;y0] 这里x0和y0代表的是 视差的搜索方向
            disp_lvl = torch.cat([x0,y0], dim=-1) 
            # 从该层的几何代价体中，利用双线性插值（bilinear_sampler）提取特征
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            # 初始相关性采样
            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + self.dx   # X on right image
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            # 相关性也通过双线性插值进行处理
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        # 实际上是在channel层进行拼接，拼接的结果是 level * 2（x&y） * (2 * radius + 1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape(B, D, H, W1)
        fmap2 = fmap2.reshape(B, D, H, W2)
        with torch.cuda.amp.autocast(enabled=False):
          corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        corr = corr.reshape(B, H, W1, 1, W2)
        # 这里实际相乘的是 [W1 D] * [D W2] = [W1 1 W2]
        # 物理意义上是宽度上每个像素点的特征相乘求余弦相似度
        return corr