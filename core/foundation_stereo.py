# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from typing import Any
import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import torchvision
from core.update import BasicSelectiveMultiUpdateBlock
from core.extractor import ContextNetDino, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import (
    BasicConv, Conv3dNormActReduced, CostVolumeDisparityAttention, FeatureAtt,
    SpatialAttentionExtractor, ChannelAttentionEnhancement, BasicConv_IN, Conv2x,
    ResnetBasicBlock3D, build_gwc_volume, build_concat_volume, disparity_regression,
    context_upsample
)
from core.utils.utils import InputPadder
# from Utils import *
import time,huggingface_hub


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    # img/255 是将像素值缩放到[0,1]之间
    # 因为 Normalize 期望输入在 [0,1] 范围内（这是 torchvision 的约定）
    # 设定通道的均值和方差，这些数值是在 ImageNet 训练集上 RGB 通道的均值和标准差（选一个比较好的初始值去泛化？）
    # .contiguous() 确保张量在内存中是连续存储的
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()

# hourglass（沙漏）对应的也就是UNet结构
class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        # 下采样
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        # 开始上采样
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1) # concat本身是在做Skip Connection的操作，避免细节的丢失
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        conv = conv + x
        conv = self.conv_out(conv)

        return conv


    ### args.hidden_dims（context_dims）
    # 直觉：这是 GRU“脑容量”的配置，告诉每层更新单元能记住多少上下文信息。
    # 不是物理量：单位是通道数（feature channels），不是米/像素。
    # 作用：
    # 通道大：表达能力强，能编码更复杂纹理/遮挡线索。
    # 通道小：速度更快、显存更省。
    # 影响：主要影响迭代更新质量（细节边缘、弱纹理区域）与成本（显存/时延）。
    ###

    ### self.cv_group = 8（group-wise correlation 分组数）
    # 直觉：把特征通道分成若干组，在每组内做相关性，类似“多视角小专家”分别投票匹配。
    # 不是物理量：是结构超参数。
    # 作用：
    # 分组少（如 1）：接近全通道相关，信息混得多。
    # 分组多：相关性更细粒度，但每组信息更窄。
    # 影响：决定代价体里“匹配相似度”的表示方式，进而影响匹配稳定性与计算开销。
    ###

    ### volume_dim = 28
    # 直觉：代价体进入 3D 聚合网络后的“内部宽度”（中间特征通道）。
    # 不是物理量：也是网络容量参数。
    # 作用：
    # 大：3D cost aggregation 能学更复杂的空间-视差模式。
    # 小：更快更省，但可能损失表达能力。
    # 影响：直接关系到 hourglass / classifier 这条 3D 路径的建模能力与计算量。
    ###

    ### args.n_gru_layers
    # 直觉：视差 refinement 用几层（多尺度）GRU来反复修正。
    # 不是物理量：是层数。
    # 作用：
    # 层数高：可在更多尺度融合上下文，修正更稳。
    # 层数低：更轻量，但长程/复杂区域修正能力可能下降。
    # 和 valid_iters 的区别：
    # n_gru_layers：每次迭代里“有几层更新器”
    # valid_iters：总共“迭代几轮”
    # 影响：决定 refinement 的层次深度与开销。
    ###

    ### args.corr_radius（以及 dx = [-r, ..., r]）
    # 最接近“几何意义”的参数。
    # 直觉：每次更新时，在当前视差附近向左右各看多远的候选位移（局部搜索窗口半径）。
    # 可理解为像素位移范围（在该层分辨率坐标系下）。
    # 作用：
    # 半径小：更新更局部，快，但可能找不到较大修正。
    # 半径大：搜索更广，能纠正更大误差，但更慢更耗内存，也更容易引入噪声匹配。
    # dx 就是把这个窗口离散化成采样偏移模板，用于构造几何相关特征。
    ###

class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims     # 分支的通道配置
        self.cv_group = 8                   # group-wise correlation 的分组数，影响代价体相关性表示
        volume_dim = 28                     # 3D 代价体后续处理中间通道数。

        # DINO后端
        # sam：spatial attention 空间注意力
        # cam：channel attention 通道注意力
        # update_block: 迭代细化视差的核心更新器（每轮输出delta）
        self.cnet = ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)     # 提供GRU迭代更新所需的上下文特征
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)  # 迭代细化视差的核心更新器（每轮输出delta）
        self.sam = SpatialAttentionExtractor()                                                                          # 空间注意力
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])                                                # 通道注意力 两个注意力用于增强context特征质量

        # n_gru_layers gru更新层数
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)]) # 暂时没有用到

        # 提取用于匹配的底层特征
        self.feature = Feature(args)                                                    # 主特征提取器，融合backbone + dino/depth-anything
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)  # 把高维特征投影到较小通道，供 concat-volume 构建 本质上是使用1*1的卷积核进行的降维

        # 对应的是1/2 1/4 尺度特征 给上采样/细化分支用
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),  # 输入通道3 输出通道32 cnn尺度 3*3 stride为2 
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),                   # 输入通道数 输出通道数 卷积核大小 步长 填充 偏置
            nn.InstanceNorm2d(32), nn.ReLU()                          # instance norm 
            )
        # 在stem_2的后面接stem_4
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        # GRU（门控循环单元）根据模型描述GRU的引入是为了进行迭代处理，
        # 原因是对于复杂的模型处理一次推理可能无法完成任务，因此需要GRU来控制每次迭代保留的信息等

        # 1/2 尺度特征 给上采样/细化分支用
        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(                                       # nn.Sequential表示按顺序执行模块
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),  # 上采样 2*32 输入通道数 9 输出通道数 4*4*9 卷积核大小 步长 填充
          )

        # 3D代价体的构建
        # 在立体匹配（Stereo Matching）中，我们会构建一个代价体积（Cost Volume）。
        # 这个体积的维度通常是：Batch * Channel * Disparity * Height * Width。
        # 3D卷积同时处理的是视差+长宽
        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),                                       # 降维处理，从32维调整到 volume_dim
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),        # 使用3*3*3的核进行空间+视差信息聚合
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1), # 引入残差结构，学习复杂的视差分布模式
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1), # 进一步加强特征，确保提取的匹配代价具有极高的辨别力，减少由于遮挡或重复纹理导致的误匹配
            )
        # 用图像特征引导代价提特征增强
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out) # 3D代价体聚合网络
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3), # // 是整除
        ) # 输出每个视差候选的代价/logit，后续softmax+回归得到初始视差

        # corr_radius 几何相关性查找的位移半径
        r = self.args.corr_radius 
        # 生成一个-r到r的等差数列，步长为1，长度为2*r+1，reshape成(1, 1, 2*r+1, 1)的形状，用来对corr_radius进行采样了
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1) # 基于 corr_radius 生成离散位移向量，用于几何编码体采样。
        self.dx = dx


    # 凸组合上采样
    # mask_feat_4: 1/4分辨率下的特征，会包含较强的语义信息
    # stem_2x： 1/2分辨率下的特征，相比之下会包含较为丰富的空间细节（比如边缘和纹理）
    # spx_2_gru： 使用空间GRU进行二者的结合，本质上是利用不同分辨率下的特征进行一轮预测
    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution 
            spx_pred = self.spx_gru(xspx)                 # 预测上采样的权重
            spx_pred = F.softmax(spx_pred, 1)             # 归一化权重，确保权重和为1
            # 执行凸组合上采样，本质上是从1/4分辨率上采样回到原图
            # 这里以输入的spx_pred权重为基础进行恢复，而非传统的线性插值
            # 传统线性插值的本质是平滑 会丢失细节
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()

    # 处理流程
    #             ┌────────────────────┐
    #             │   image1, image2   │
    #             └──────────┬─────────┘
    #                        │
    #                 normalize_image
    #                        │
    #                        ▼
    #             ┌────────────────────┐
    #             │  Shared Backbone   │
    #             │  CNN + ViT         │
    #             └──────────┬─────────┘
    #                        │
    #          ┌─────────────┴─────────────┐
    #          ▼                           ▼
    #   features_left              features_right
    #          │                           │
    #          └──────────┬────────────────┘
    #                     ▼
    #           ┌────────────────┐
    #           │  Cost Volume   │
    #           │  GWC + Concat  │
    #           └────────┬───────┘
    #                    ▼
    #           3D CNN Hourglass
    #                    ▼
    #           Aggregated Volume
    #                    ▼
    #            Classifier + Softmax
    #                    ▼
    #          Soft Argmin (Init Disp)
    #                    │
    #                    ▼
    #     ┌────────────────────────────────┐
    #     │        GRU Iterations          │
    #     │                                │
    #     │  geo_fn(disp) → geo_feat       │
    #     │         ↓                      │
    #     │  update_block (ConvGRU)        │
    #     │         ↓                      │
    #     │  disp = disp + delta_disp      │
    #     └────────────────────────────────┘
    #                    │
    #                    ▼
    #            Learned Upsampling
    #                    ▼
    #             Final Disparity
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        """ Estimate disparity between pair of frames """
        B = len(image1)
        low_memory = low_memory or (self.args.get('low_memory', False))
        # 输入两个图像
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        # autocast自动混合精度
        with autocast(enabled=self.args.mixed_precision):
            # 一次性提取出两幅图像的特征
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            # 恢复到两个图
            vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            # 降采样到1/2的特征上
            stem_2x = self.stem_2(image1)

            # 构建代价volume
            gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
            # 聚合特征通道 & 执行拼接
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4) # 得到拼接之后的代价
            del left_tmp, right_tmp # 删除掉无用的

            # 将gwc_volume与特征通道聚合之后的代价再次进行拼接
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left) # hourglass聚合
            # 经过多层推理网络之后得到聚合的特整理

            # Init disp from geometry encoding volume
            # comb_volume是前面推理之后的结果，对应的维度是[B C D H W](具体维度需要再查查代码确认)
            # classifier自定义的分类器
            # 再经过softmax转到对应的概率进行后续处理
            # squeeze(1)是去掉维度为1的维度，这里去掉的是通道维度，得到[B, max_disp, H, W]
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  #(B, max_disp, H, W)

            # 通过disparity_regression回归的方式初始化双目视差
            if init_disp is None:
              init_disp = disparity_regression(prob, self.args.max_disp//4)  # Weighted  sum of disparity

            # 提取多尺度的特征信息 这里只对左图提取特征
            cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list[Any](cnet_list)
            # 分别使用tanh和relu来对输入特征进行处理
            net_list = [torch.tanh(x[0]) for x in cnet_list]   # Hidden information
            inp_list = [torch.relu(x[1]) for x in cnet_list]   # Context information list of pyramid levels
            inp_list = [self.cam(x) * x for x in inp_list]     # 通道注意力用来表征哪些特征对当前更重要
            att = [self.sam(x) for x in inp_list]              # 空间注意力用来表征哪些区域对当前更重要

        # 借鉴RAFT的思想通过corse to fine的方式逐渐恢复
        # Geometric encoding 代价构建实例
        geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
        b, c, h, w = features_left[0].shape
        # 这行代码创建了一个 仅包含水平像素坐标的张量，形状 (B, H, W, 1)，其中每个位置的值等于其列索引（x 坐标），用于后续的空间变换或几何计算。
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        disp_preds = []

        # GRUs iterations to update disparity (1/4 resolution)
        # 使用GRU进行迭代优化
        for itr in range(iters):
            disp = disp.detach() # 每次 GRU 更新只对当前步负责 不做跨iteration的处理 
            # 初始视差、坐标、代价体作为输入进行forwar
            geo_feat = geo_fn(disp, coords, low_memory=low_memory) # 根据当前视差在右特征图里面找到对应的位置
            # 更新GRU网络 并修算出修正量
            with autocast(enabled=self.args.mixed_precision):
              net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            # 执行修正
            disp = disp + delta_disp.float()
            if test_mode and itr < iters-1:
                continue

            # 在最后一次修正之后执行上采样恢复图像
            # upsample predictions
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)


        if test_mode:
            return disp_up

        return init_disp, disp_preds


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
      B,_,H,W = image1.shape
      # 缩小图像
      img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      # 执行padding对齐
      padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
      img1_small, img2_small = padder.pad(img1_small, img2_small) 
      # 执行单尺度推理
      disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
      # 移除padding
      disp_small = padder.unpad(disp_small.float())
      # 上采样到原始大小（1/small_ratio）使用的是插值的方式
      disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio # 
      # 裁剪到0-1之间 保证非负视差
      disp_small_up = disp_small_up.clip(0, None)
      # 创建原尺寸的padder
      padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
      image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up) # 对原图和视差图一起padding
      # 平移视差，避免padding的影响
      disp_small_up += padder._pad[0]
      # 视差降到1/4是为了对齐尺度变化，原因是代码内部是在1/4尺度上处理的
      init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
      disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
      disp = padder.unpad(disp.float()) # 移除padding
      return disp

