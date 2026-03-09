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
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# 表征是否是连续存储
def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting(): # 检查当前代码是否正在被TorchScript 编译器处理
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

# 2D LayerNorm
class LayerNorm2d(nn.LayerNorm):
    r""" https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py#L85
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        @x: (B,C,H,W)
        """
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        else:
            # 求norm的公式
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


# 统一封装 2D/3D 卷积（或反卷积） + 可选归一化 + 可选激活
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm3d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm2d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

# 把一次完整 "3D 卷积"拆成两步 ，降低计算复杂度
# 分成全量卷积还是只在视差维度上面做卷积
class Conv3dNormActReduced(nn.Module):
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
          kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1,kernel_size,kernel_size), padding=(0, kernel_size//2, kernel_size//2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        # 这是只在视差维度上面做卷积
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        @x: (B,C,D,H,W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x



# **作用**：2D/3D 残差块。  
# 公式直觉：`out = ReLU(F(x) + shortcut(x))`
class ResnetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out

# 与ResnetBasicBlock类似
class ResnetBasicBlock3D(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out

# 多头注意力
class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads              # 头的数量
        self.embed_dim = embed_dim              # 输入输出特征总维度
        self.head_dim = embed_dim // num_heads  # 头的维度
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # 线性投影矩阵，实际上是要学习的参数
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        # 投影之后的结果
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 张量重排，以符合各个头自己进行隔离的目的
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = F.scaled_dot_product_attention(Q, K, V) # 注意力计算  Q,K,V 都是 (B,L,num_heads,head_dim)   

        
        attn_output = attn_output.reshape(B,L,-1)           
        output = self.out_proj(attn_output) # 全连接输出

        return output 



class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # 注意这里的dropout是训练时候使用的，推理模式下会关闭
    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size) # 自注意力层
        src = src + self.dropout1(src2)                                         # 第一条残差链接
        src = self.norm1(src)                                                   # 第一层归一化

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))          # 前馈FFN层
        src = src + self.dropout2(src2)                                         # 第二条残差链接 
        src = self.norm2(src)

        return src


# 上采样插值
class UpsampleConv(nn.Module):
    def __init__(self, C_in, C_out, is_3d=False, kernel_size=3, bias=True, stride=1, padding=1):
        super().__init__()
        self.is_3d = is_3d
        if is_3d:
          self.conv = nn.Conv3d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        else:
          self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        if self.is_3d:
          mode = 'trilinear'
        else:
          mode = 'bilinear'
        x = F.interpolate(x, size=None, scale_factor=2, align_corners=False, mode=mode)
        x = self.conv(x)
        return x


# Unet风格的上采样
class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

# 使用InstanceNorm的卷积
class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

# 使用InstanceNorm的Unet风格上采样
class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(out_channels*2, out_channels*mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

# 通道分组之间的相关性计算
def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0, f"C:{C}, num_groups:{num_groups}"
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    with torch.cuda.amp.autocast(enabled=False):
      cost = (F.normalize(fea1.float(), dim=2) * F.normalize(fea2.float(), dim=2)).sum(dim=2)  #!NOTE Divide first for numerical stability
    assert cost.shape == (B, num_groups, H, W)
    return cost

# 构建GWC代价体
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """
    @refimg_fea: left image feature
    @targetimg_fea: right image feature
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


# 构建拼接型代价体
def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


# d代表视差 P(d) 对应视差为d的概率
# d*P(d) 对应计算的是期望
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

# 使用2D特征去调制3D代价体
# 利用原始图像的 2D 特征（Context）来“过滤”或“增强” 3D 代价体积（Cost Volume）中的信息。
class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1)
            )

    def forward(self, cv, feat):
        '''
        @cv: cost volume (B,C,D,H,W)
        @feat: (B,C,H,W)
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)   #(B,C,1,H,W)
        # 输入 feat: (B, feat_chan, H, W)
        # 输出 feat_att: (B, cv_chan, H, W) 
        # 这里通过两个卷积层，将 2D 图像特征的通道数转换到与代价体积（Cost Volume）的通道数 $C$ 一致。
        # 这相当于在学习一种“注意力图”，决定每一个通道在不同空间位置的重要性。
        # 而后执行对齐
        cv = torch.sigmoid(feat_att)*cv # 对输出进行加权
        return cv

# 凸组合上采样
def context_upsample(disp_low, up_weights):
    """
    @disp_low: (b,1,h,w)  1/4 resolution
    @up_weights: (b,9,4*h,4*w)  Image resolution
    """
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)

    return disp


# 位置编码
class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe
    # self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]


# 对 cost volume 的 disparity 维做 Transformer 注意力
class CostVolumeDisparityAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
        super().__init__()
        self.resize_embed = resize_embed
        self.sa = nn.ModuleList([])
        for _ in range(num_transformer):
        self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
        self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)

    # Cost Volume
    # (B,C,D,H,W)
    #       │
    # reshape
    #       │
    # (BHW,D,C)
    #       │
    # Positional Embedding
    #       │
    # Transformer × 6
    #       │
    # reshape back
    #       │
    # (B,C,D,H,W)
    def forward(self, cv, window_size=(-1,-1)):
        """
        @cv: (B,C,D,H,W) where D is max disparity
        """
        x = cv
        B,C,D,H,W = x.shape
        x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C) # 将代价体变形为 (B*H*W, D, C)
        # 加位置编码
        x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
        for i in range(len(self.sa)):
            x = self.sa[i](x, window_size=window_size) # Transformer encoder
        x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2) # reshape

        return x


# 通道增强
class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# CBAM的空间注意力 CBAM 风格注意力（Channel + Spatial）
# CBAM (Convolutional Block Attention Module) 论文中的 通道注意力模块 (Channel Attention Module)
class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)



class EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm=='layer':
          self.norm = LayerNorm2d(dim, eps=1e-6)
        else:
          self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU() # 使用 GELU 激活函数，这比 ReLU 更平滑，是目前高性能视觉模型的标配
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x