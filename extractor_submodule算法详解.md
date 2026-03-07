# FoundationStereo `extractor.py` 与 `submodule.py` 算法详解（零基础版）

这份文档面向没有 Python / PyTorch 基础的同学，目标是：

- 看懂这两个文件里“每个函数/类大概做什么”
- 知道常见参数是什么意思
- 知道输入输出张量（tensor）形状怎么变化
- 明白它们在双目立体匹配（stereo）中的位置

---

## 0. 先补 5 个最小基础概念

### 0.1 Tensor 是什么

在 PyTorch 里，图像和特征都是 `tensor`（多维数组）。

- 图像常见形状：`(B, C, H, W)`
- `B`：batch size，一次处理几张图
- `C`：通道数（RGB 图像通常是 3）
- `H/W`：高和宽

### 0.2 卷积层是什么

`Conv2d` / `Conv3d` 可以理解成“局部窗口提取特征”。

- `stride=2`：通常把分辨率减半（下采样）
- `ConvTranspose` 或插值 + 卷积：通常把分辨率放大（上采样）

### 0.3 归一化层是什么

常见有 `BatchNorm`、`InstanceNorm`、`LayerNorm`。作用是稳定训练、加快收敛。

### 0.4 残差连接（Residual）

`out = x + F(x)`：保留原信息，让深网络更容易训练。

### 0.5 代价体（Cost Volume）

双目匹配里核心是：给左图每个像素，枚举一系列视差 `d`，看右图“向左平移 d”后有多像。  
把“像不像”的结果堆起来，就是代价体，常见形状 `(B, C, D, H, W)`。

- `D`：候选视差数量（disparity levels）

---

## 1. `core/extractor.py` 详解

这个文件主要负责**特征提取**，包括：

- CNN 特征（EdgeNeXt）
- ViT/DepthAnything 语义特征
- 给后续 GRU 更新器用的上下文特征

---

### 1.1 `ResidualBlock`

#### 作用

标准残差块：两层卷积 + 归一化 + ReLU，再和输入相加。

#### 构造参数（`__init__`）

- `in_planes`：输入通道数
- `planes`：输出通道数
- `norm_fn`：归一化类型，可选 `group/batch/instance/layer/none`
- `stride`：第一层卷积步长，`2` 常用于降采样

#### 前向参数（`forward(x)`）

- `x`：`(B, in_planes, H, W)`

#### 输出

- `(B, planes, H', W')`
- 当 `stride=2` 时，`H'/W'` 通常减半

---

### 1.2 `MultiBasicEncoder`

#### 作用

多尺度 CNN 编码器，输出 1/4、1/8、1/16 三个尺度特征（代码里对应 `outputs04/08/16`）。

#### 构造参数

- `output_dim`：每个输出头的通道配置列表；每个元素像 `[dim16, dim08, dim04]`
- `norm_fn`：归一化类型
- `dropout`：dropout 概率
- `downsample`：控制前几层是否使用步长 2

#### 关键方法

- `_make_layer(dim, stride)`：创建两个 `ResidualBlock` 组成的小 stage

#### 前向参数（`forward(x, dual_inp=False, num_layers=3)`）

- `x`：`(B,3,H,W)`
- `dual_inp`：若 `True`，输入通常是左右图拼在 batch 维，内部会拆分一部分特征
- `num_layers`：返回几层输出，`1/2/3`

#### 输出

- `num_layers=3` 时返回 `(outputs04, outputs08, outputs16)`（或附加 `v`）
- `outputs04` 中每个特征是约 1/4 分辨率

---

### 1.3 `ContextNetDino`

> 继承 `MultiBasicEncoder` 思路，但在 forward 里融合了 ViT 特征。

#### 作用

为更新模块提供上下文特征，并把 `vit_feat`（来自 DepthAnything）注入到 1/4 尺度特征。

#### 构造参数

- `args`：全局配置，关键字段是 `args.vit_size`（`vits/vitb/vitl`）
- `output_dim`：多尺度输出通道配置
- `norm_fn`：归一化类型
- `downsample`：下采样控制

#### 前向参数（`forward(x_in, vit_feat, dual_inp=False, num_layers=3)`）

- `x_in`：输入图像 `(B,3,H,W)`
- `vit_feat`：ViT 特征（已对齐到约 1/4 尺度）

#### 输出

- `(outputs04, outputs08, outputs16)` 三个尺度特征

#### 备注

- 代码中有 `H_resize/W_resize` 的计算，但在这个 forward 里并未继续使用。

---

### 1.4 `DepthAnythingFeature`

#### 作用

封装 DepthAnything 模型，提取多层中间特征，并输出 `depth_head` 的中间路径特征。

#### 构造参数

- `encoder`：`vits/vitb/vitl`，决定 ViT 容量和通道维度

#### 前向参数（`forward(x)`）

- `x`：`(B,C,H,W)`，通常是 resize 后输入

#### 输出（字典）

- `out`：主输出特征
- `path_1/path_2/path_3/path_4`：多尺度中间特征
- `features`：ViT 中间层 token 特征
- `disp`：Depth head 相关输出

---

### 1.5 `Feature`

#### 作用

这是主特征提取器：把 EdgeNeXt CNN 特征和 DepthAnything 语义特征融合，输出多尺度特征列表。

#### 构造参数

- `args`：主要用 `args.vit_size`

#### 结构流程

1. 用 `timm` 创建 `edgenext_small`
2. 用 `DepthAnythingFeature` 提取 ViT 特征（并 `freeze_model` 冻结参数）
3. CNN backbone 得到 `x4/x8/x16/x32`
4. 通过 `Conv2x_IN` 连续上采样并和浅层特征拼接融合
5. 在 1/4 尺度把 `vit_feat` 拼到 `x4` 并卷积融合

#### 前向参数（`forward(x)`）

- `x`：图像 `(B,C,H,W)`

#### 输出

- `([x4, x8, x16, x32], vit_feat)`
- 第一项是多尺度特征列表，第二项是 ViT 特征（用于 context 网络）

---

## 2. `core/submodule.py` 详解

这个文件是“通用积木库”：卷积块、注意力、代价体构建、视差回归、上采样等。

---

### 2.1 `_is_contiguous(tensor)`

#### 作用

检查 tensor 内存是否连续（用于优化 `LayerNorm2d` 分支）。

#### 参数与返回

- 参数：`tensor`
- 返回：`bool`

---

### 2.2 `LayerNorm2d`

#### 作用

给 `(B,C,H,W)` 特征做 LayerNorm（按通道归一化），兼容 channels-first。

#### 前向参数

- `x`：`(B,C,H,W)`

#### 输出

- 与输入同形状

---

### 2.3 `BasicConv`

#### 作用

统一封装 2D/3D 卷积或反卷积 + 可选归一化 + 可选激活。

#### 构造参数

- `in_channels/out_channels`
- `deconv`：是否用反卷积
- `is_3d`：是否 3D 卷积
- `bn`：是否使用归一化
- `relu`：是否激活
- `norm`：`batch` 或 `instance`
- `**kwargs`：传给卷积的 `kernel_size/stride/padding` 等

#### 前向

- 输入输出形状取决于卷积配置

---

### 2.4 `Conv3dNormActReduced`

#### 作用

把 3D 卷积分解成两步：

- 先在空间维 `(H,W)` 卷积
- 再在视差维 `D` 卷积

这样计算更省一些。

#### 构造参数

- `C_in/C_out`：输入输出通道
- `hidden`：中间通道（默认等于 `C_out`）
- `kernel_size`：空间卷积核
- `kernel_disp`：视差维卷积核
- `stride`
- `norm`：3D 归一化层类型

#### 前向

- 输入：`(B,C,D,H,W)`
- 输出：`(B,C_out,D',H',W')`

---

### 2.5 `ResnetBasicBlock` / `ResnetBasicBlock3D`

#### 作用

2D/3D 版本残差块。

#### 参数要点

- `inplanes/planes`：输入输出通道
- `kernel_size/stride/padding`
- `downsample`：当尺寸或通道不匹配时对 shortcut 做投影
- `norm_layer`：归一化层类型

---

### 2.6 `FlashMultiheadAttention`

#### 作用

多头注意力层，内部调用 `F.scaled_dot_product_attention`（PyTorch 高效实现）。

#### 前向参数

- `query/key/value`：形状 `(B,L,C)`
- `attn_mask/window_size`：接口保留，当前实现主要走全局注意力

#### 输出

- `(B,L,C)`

---

### 2.7 `FlashAttentionTransformerEncoderLayer`

#### 作用

标准 Transformer Encoder 一层：

- Self-Attention
- FeedForward
- 两次残差 + LayerNorm

#### 前向参数

- `src`：`(B,L,C)`

#### 输出

- `(B,L,C)`

---

### 2.8 `UpsampleConv`

#### 作用

先插值放大 2 倍，再卷积融合（可 2D/3D）。

#### 参数

- `C_in/C_out`
- `is_3d`
- `kernel_size/bias/stride/padding`

---

### 2.9 `Conv2x`

#### 作用

U-Net 风格上采样融合块：

1. `conv1` 把特征上采样到高分辨率
2. 与 skip 特征 `rem` 拼接（或相加）
3. `conv2` 再融合

#### 关键参数

- `deconv`：是否用反卷积上采样
- `is_3d`：2D/3D
- `concat`：与 skip 拼接还是相加
- `keep_concat`：拼接后输出通道是否保持翻倍
- `keep_dispc`：3D 场景下是否保持 disparity 维不变

#### 前向参数

- `x`：低分辨率输入
- `rem`：同层 skip 特征

---

### 2.10 `BasicConv_IN` / `Conv2x_IN`

#### 作用

与上面类似，但默认走 `InstanceNorm`，常用于图像风格/域变化更强时。

`Feature` 模块里就大量用了 `Conv2x_IN`。

---

### 2.11 `groupwise_correlation(fea1, fea2, num_groups)`

#### 作用（非常关键）

把通道分组后做相关性（相似度）：

1. 通道 `C` 分成 `num_groups` 组
2. 每组做归一化点积
3. 得到每组一个相关性图

#### 参数

- `fea1/fea2`：左右特征，形状 `(B,C,H,W)`
- `num_groups`：分组数，要求 `C % num_groups == 0`

#### 输出

- `(B, num_groups, H, W)`

---

### 2.12 `build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1)`

#### 作用（非常关键）

构建 GWC（Group-wise Correlation）代价体。

核心逻辑：对每个候选视差 `i`，把右特征向左偏移 `i`，再和左特征做 `groupwise_correlation`。

#### 参数

- `refimg_fea`：左图特征 `(B,C,H,W)`
- `targetimg_fea`：右图特征 `(B,C,H,W)`
- `maxdisp`：最大候选视差数量（离散层数）
- `num_groups`：分组数
- `stride`：接口保留，当前函数体里未使用

#### 输出

- `volume`：`(B, num_groups, maxdisp, H, W)`

---

### 2.13 `build_concat_volume(refimg_fea, targetimg_fea, maxdisp)`

#### 作用

构建另一种代价体：不是相关性，而是直接把左右特征拼接后沿 `D` 堆叠。

#### 参数

- 左右特征 `(B,C,H,W)`
- `maxdisp`

#### 输出

- `(B, 2C, maxdisp, H, W)`

---

### 2.14 `disparity_regression(x, maxdisp)`

#### 作用（非常关键）

把“每个视差层的概率”转成“期望视差值”。

数学上：`disp = sum_{d=0}^{maxdisp-1} p(d) * d`

#### 参数

- `x`：通常是 softmax 后概率，形状 `(B, maxdisp, H, W)`
- `maxdisp`：视差层数

#### 输出

- `(B,1,H,W)`，连续值视差图

---

### 2.15 `FeatureAtt`

#### 作用

用 2D 特征 `feat` 生成注意力权重，去调制 3D 代价体 `cv`。

#### 前向参数

- `cv`：`(B,C,D,H,W)`
- `feat`：`(B,C_feat,H,W)`

#### 输出

- 与 `cv` 同形状，但被 `sigmoid(att)` 加权

---

### 2.16 `context_upsample(disp_low, up_weights)`

#### 作用（非常关键）

把 1/4 分辨率视差图上采样到原图分辨率，方式是**学习到的 3x3 邻域加权重建**，不是普通双线性插值。

#### 参数

- `disp_low`：`(B,1,h,w)`，低分辨率视差
- `up_weights`：`(B,9,4h,4w)`，每个高分辨率像素对应 3x3 的 9 个权重

#### 输出

- `(B,4h,4w)` 的高分辨率视差

---

### 2.17 `PositionalEmbedding`

#### 作用

标准正弦余弦位置编码，给序列特征加位置信息。

#### 前向参数

- `x`：`(B,N,D)`（序列长度 `N`，特征维 `D`）
- `resize_embed`：当 `N` 超过预设长度时是否线性插值位置编码

#### 输出

- `(B,N,D)`

---

### 2.18 `CostVolumeDisparityAttention`

#### 作用（关键）

在代价体的“视差维 D”上做 Transformer 注意力，增强不同视差层之间的关系建模。

#### 输入输出

- 输入：`cv` `(B,C,D,H,W)`
- 内部 reshape 为 `(B*H*W, D, C)`，把每个像素位置看作一个长度为 `D` 的序列
- 输出：再 reshape 回 `(B,C,D,H,W)`

#### 直觉

它在问：**“这个像素在不同候选视差之间，哪些层互相支持？”**

---

### 2.19 `ChannelAttentionEnhancement`

#### 作用

通道注意力（类似 CBAM 的 channel 分支）：

- 全局平均池化 + 全局最大池化
- 两条路共享 MLP（1x1 conv 实现）
- 相加后 sigmoid，得到每个通道权重

---

### 2.20 `SpatialAttentionExtractor`

#### 作用

空间注意力（类似 CBAM 的 spatial 分支）：

- 对通道做 mean 和 max，得到 2 张图
- 拼接后做卷积 + sigmoid
- 得到每个像素位置权重

---

### 2.21 `EdgeNextConvEncoder`

#### 作用

EdgeNeXt 风格卷积编码块：

- depthwise conv（逐通道空间卷积）
- pointwise MLP（线性层扩张再压回）
- 残差连接

#### 参数

- `dim`：通道维
- `layer_scale_init_value`：残差分支缩放参数初值
- `expan_ratio`：MLP 扩张倍率
- `kernel_size`
- `norm`

---

## 3. 这两个文件在完整算法里的分工

可以把它们理解为：

- `extractor.py`：负责“把图像变成高质量多尺度特征”
- `submodule.py`：提供“匹配和聚合需要的基础算子”

在 `FoundationStereo` 主流程中（你前面看的 `foundation_stereo.py`）：

1. `Feature` 产出左右图多尺度特征 + `vit_feat`
2. `build_gwc_volume/build_concat_volume` 构建代价体
3. `FeatureAtt`、`CostVolumeDisparityAttention` 等增强代价体表达
4. `disparity_regression` 得到初始视差
5. `context_upsample` 把低分辨率结果恢复到高分辨率

---

## 4. 给零基础同学的阅读建议（非常实用）

按下面顺序读，效率最高：

1. `Feature.forward`（先看输入输出）
2. `groupwise_correlation`（理解“相似度”）
3. `build_gwc_volume`（理解“枚举视差”）
4. `disparity_regression`（理解“概率转视差”）
5. `context_upsample`（理解“学习型上采样”）
6. 最后再看 `CostVolumeDisparityAttention`（Transformer 加强版）

---

## 5. 常见参数速查表

- `in_channels/out_channels`：输入/输出通道数
- `kernel_size`：卷积核大小
- `stride`：步长（2 常表示降采样）
- `padding`：边界补零
- `deconv=True`：反卷积上采样
- `is_3d=True`：处理 3D 代价体 `(D,H,W)`
- `num_groups`：分组相关性里的组数
- `maxdisp`：最大候选视差数（离散层数）
- `norm_fn/norm`：归一化类型（batch/instance/layer/group）

---

## 6. 你可以立刻做的一个小实验

在主 `forward` 里打几个 shape log（只要 5 行）：

- `x4/x8/x16/x32`
- `gwc_volume.shape`
- `concat_volume.shape`
- `prob.shape`（softmax 后）
- `init_disp.shape`

你会很快把“抽象算法”变成“可见的数据流”。

