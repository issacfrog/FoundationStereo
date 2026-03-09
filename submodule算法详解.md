# FoundationStereo `core/submodule.py` 算法详解（零基础友好版）

这份文档专门讲 `core/submodule.py`。  
目标：你不需要先会 PyTorch，也能读懂“每个模块做什么、参数是什么意思、在立体匹配流程里扮演什么角色”。

---

## 1. 这个文件在整个项目里负责什么

`submodule.py` 不是主网络入口，而是“工具箱 + 积木库”。它主要提供四类能力：

- **基础神经网络积木**：卷积块、残差块、上采样块、归一化块
- **代价体构建核心**：`groupwise_correlation`、`build_gwc_volume`、`build_concat_volume`
- **视差解码与上采样**：`disparity_regression`、`context_upsample`
- **注意力增强模块**：通道注意力、空间注意力、视差维 Transformer 注意力

你可以把它理解成：  
`foundation_stereo.py` 在“拼装流程”，而 `submodule.py` 在“提供可复用零件”。

---

## 2. 先看 7 个最重要函数/类（建议优先读）

如果时间有限，先读这些：

1. `groupwise_correlation`
2. `build_gwc_volume`
3. `build_concat_volume`
4. `disparity_regression`
5. `context_upsample`
6. `CostVolumeDisparityAttention`
7. `Conv2x` / `Conv2x_IN`

它们覆盖了“匹配 -> 代价体 -> 初始视差 -> 上采样”的主链路。

---

## 3. Tensor 形状速记（后面会反复用）

- 图像/特征常见：`(B, C, H, W)`
- 代价体常见：`(B, C, D, H, W)`
- 含义：
  - `B`：batch size
  - `C`：通道数
  - `H/W`：空间高宽
  - `D`：候选视差层数（max disparity bins）

---

## 4. 逐模块详解

---

### 4.1 `_is_contiguous(tensor)`

**作用**：检查内存是否连续，用于 `LayerNorm2d` 的分支优化。  
**输入**：任意 tensor。  
**输出**：`True/False`。

---

### 4.2 `LayerNorm2d`

**作用**：给 `channels_first` 形式 `(B,C,H,W)` 做 LayerNorm。  

为什么要自定义：PyTorch 原生 LayerNorm更常见于最后一维（如 `(B,N,D)`），这里通过 `permute` 转换到 `(B,H,W,C)` 再归一化，最后转回。

**forward(x)**：

- 输入：`(B,C,H,W)`
- 输出：同形状

---

### 4.3 `BasicConv`

**作用**：统一封装 2D/3D 卷积（或反卷积） + 可选归一化 + 可选激活。

**常用参数**：

- `in_channels/out_channels`：输入输出通道
- `deconv`：`True` 表示反卷积（常用于上采样）
- `is_3d`：`True` 表示处理 3D 数据（代价体）
- `bn`：是否用归一化层
- `norm`：`batch` 或 `instance`
- `relu`：是否加 LeakyReLU
- `kernel_size/stride/padding`：通过 `**kwargs` 传入

---

### 4.4 `Conv3dNormActReduced`

**作用**：把一次“完整 3D 卷积”拆成两步，降低计算复杂度：

1. 在空间维 `(H,W)` 上卷积：`kernel=(1,k,k)`
2. 在视差维 `D` 上卷积：`kernel=(k_disp,1,1)`

**输入输出**：

- 输入：`(B,C,D,H,W)`
- 输出：`(B,C_out,D',H',W')`

这种分解在代价体网络里很常见，速度和显存更友好。

---

### 4.5 `ResnetBasicBlock` / `ResnetBasicBlock3D`

**作用**：2D/3D 残差块。  
公式直觉：`out = ReLU(F(x) + shortcut(x))`

**关键参数**：

- `inplanes/planes`：输入输出通道
- `norm_layer`：归一化层类型（可传 `BatchNorm`、`InstanceNorm`）
- `downsample`：当通道或尺寸不匹配时，对 shortcut 投影对齐

---

### 4.6 `FlashMultiheadAttention`

**作用**：多头注意力，内部调用 `F.scaled_dot_product_attention`（高效实现）。

**输入**：

- `query/key/value`：`(B,L,C)`
  - `L` 是序列长度
  - `C` 是特征维

**输出**：

- `(B,L,C)`

在本文件里，它主要服务于“视差维序列建模”（后面的 `CostVolumeDisparityAttention`）。

---

### 4.7 `FlashAttentionTransformerEncoderLayer`

**作用**：标准 Transformer Encoder 层：

- Self-Attention
- FFN（前馈网络）
- 残差连接 + LayerNorm

**输入输出**：都是 `(B,L,C)`。

---

### 4.8 `UpsampleConv`

**作用**：先插值上采样 2 倍，再卷积融合。  
支持 2D（bilinear）和 3D（trilinear）。

---

### 4.9 `Conv2x`

**作用**：典型 U-Net 风格“上采样 + skip 融合”模块。

`forward(x, rem)` 的流程：

1. `conv1`：把低分辨率特征 `x` 放大（通常 2 倍）
2. 若尺寸不一致，插值对齐到 `rem` 尺寸
3. 与 skip 特征 `rem` 融合（拼接或相加）
4. `conv2`：再卷积融合输出

**关键参数**：

- `concat=True`：和 skip 按通道拼接（常用）
- `keep_concat`：拼接后是否保持通道翻倍
- `keep_dispc`：3D 场景时是否保持 disparity 维不下采样

---

### 4.10 `BasicConv_IN` 与 `Conv2x_IN`

**作用**：和 `BasicConv/Conv2x` 同类，但默认更偏向 InstanceNorm。  
`Feature` 提取器里大量用的是这套 `_IN` 版本。

---

## 5. 代价体构建三件套（核心中的核心）

---

### 5.1 `groupwise_correlation(fea1, fea2, num_groups)`

**功能**：按通道分组做相关性（相似度）计算。  

**输入**：

- `fea1`：左图特征 `(B,C,H,W)`
- `fea2`：右图特征 `(B,C,H,W)`
- `num_groups`：分组数，要求 `C % num_groups == 0`

**步骤**：

1. 把通道 `C` 拆成 `num_groups` 组
2. 每组内先做 `L2 normalize`（提高数值稳定性）
3. 逐组点积并沿组内通道求和

**输出**：

- `(B, num_groups, H, W)`：每组一张相关性图

**直觉**：不是“全通道一次性匹配”，而是“分组匹配”，更细粒度、更稳。

---

### 5.2 `build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1)`

**功能**：构建 GWC 代价体（Group-wise Correlation Volume）。

**输入**：

- 左特征 `refimg_fea`：`(B,C,H,W)`
- 右特征 `targetimg_fea`：`(B,C,H,W)`
- `maxdisp`：候选视差层数
- `num_groups`：分组数

**关键循环**：

- 对每个候选视差 `i=0...maxdisp-1`：
  - 右图特征向左偏移 `i`
  - 与左图对应区域做 `groupwise_correlation`
  - 填进 `volume[:,:,i,:,:]`

**输出**：

- `volume` 形状 `(B, num_groups, maxdisp, H, W)`

**为什么关键**：这一步直接定义了“每个像素在每个视差候选下有多匹配”。

---

### 5.3 `build_concat_volume(refimg_fea, targetimg_fea, maxdisp)`

**功能**：构建拼接型代价体（Concat Volume），不是相关性而是“堆叠左右特征”。

**输入**：

- 左右特征 `(B,C,H,W)`
- `maxdisp`

**输出**：

- `(B,2C,maxdisp,H,W)`

它和 GWC 常被一起使用：一个显式相关性，一个保留原始描述子信息。

---

## 6. 从概率到视差：`disparity_regression`

### 6.1 `disparity_regression(x, maxdisp)`

**功能**：把“离散视差概率分布”变成“连续视差值”。

**输入**：

- `x`：通常是 softmax 后概率，形状 `(B,maxdisp,H,W)`

**公式**：

`disp = sum_{d=0}^{maxdisp-1} p(d) * d`

**输出**：

- `(B,1,H,W)`

这就是经典的 soft-argmin 风格回归。

---

## 7. 代价体增强与注意力模块

---

### 7.1 `FeatureAtt`

**功能**：让 2D 特征去调制 3D 代价体。

- 用 `feat` 生成注意力图 `feat_att`
- `sigmoid(feat_att)` 后与 `cv` 相乘

**输入**：

- `cv`: `(B,C,D,H,W)`
- `feat`: `(B,C_feat,H,W)`

**输出**：

- `(B,C,D,H,W)`（被加权后的代价体）

---

### 7.2 `PositionalEmbedding`

**功能**：标准正弦余弦位置编码，给序列加位置信息。  
在这里是给“视差序列”用。

**输入**：`(B,N,D)`  
**输出**：`(B,N,D)`

---

### 7.3 `CostVolumeDisparityAttention`

**功能**：对代价体的“视差维 D”做 Transformer 编码。

**输入**：`cv` `(B,C,D,H,W)`  
内部会变形为：`(B*H*W, D, C)`

含义是：  
把每个像素位置看成一个长度为 `D` 的序列，学习“不同视差候选之间的依赖关系”。

**输出**：再 reshape 回 `(B,C,D,H,W)`。

---

### 7.4 `ChannelAttentionEnhancement`

**功能**：通道注意力（类似 CBAM channel 分支）：

- 全局平均池化 + 全局最大池化
- 共享小 MLP（1x1 conv 实现）
- sigmoid 得到每个通道权重

---

### 7.5 `SpatialAttentionExtractor`

**功能**：空间注意力（类似 CBAM spatial 分支）：

- 通道维做平均和最大，得到 2 通道 map
- 卷积融合后 sigmoid
- 输出每个像素位置权重图

---

## 8. 上采样关键：`context_upsample`

### 8.1 `context_upsample(disp_low, up_weights)`

**功能**：把低分辨率视差（通常 1/4）恢复到高分辨率。

它不是普通插值，而是“**学习到的 3x3 邻域加权重建**”：

1. `F.unfold` 取出低分辨率视差每个位置的 3x3 邻域（9 个值）
2. 用 nearest 把这 9 通道扩展到高分辨率
3. 与 `up_weights (B,9,4h,4w)` 点乘求和

**输入**：

- `disp_low`: `(B,1,h,w)`
- `up_weights`: `(B,9,4h,4w)`

**输出**：

- `(B,4h,4w)`（高分辨率视差）

---

## 9. `EdgeNextConvEncoder`

**功能**：EdgeNeXt 风格的轻量卷积编码块：

- `dwconv`：depthwise 卷积（每个通道单独卷）
- `pwconv1/pwconv2`：通道混合 MLP（扩张再压回）
- 残差连接：`x = input + x`

是一个兼顾效率和表达能力的局部特征增强块。

---

## 10. 把整个 `submodule.py` 串成一条算法线

可以这样记：

1. 用 `BasicConv/ResBlock/Conv2x` 这类积木搭建网络子结构
2. 用 `build_gwc_volume + build_concat_volume` 构建代价体
3. 用 `FeatureAtt` 与 `CostVolumeDisparityAttention` 增强代价体表达
4. 用 `disparity_regression` 得到初始视差
5. 用 `context_upsample` 把低分辨率视差恢复到高分辨率

`submodule.py` 本质是“立体匹配里最常见操作的实现集合”。

---

## 11. 参数速查（初学者常问）

- `maxdisp`：候选视差层数（不是最终像素单位的最大深度）
- `num_groups`：相关性分组数（越大每组越细，但开销与表达会变化）
- `is_3d=True`：处理代价体而非2D图像
- `deconv=True`：使用反卷积进行上采样
- `keep_dispc=True`：3D 场景下尽量保持 disparity 维长度
- `concat=True`：skip 连接时拼接而不是相加

---

## 12. 学习时最容易踩的坑

- **形状错位**：`(B,C,H,W)` 和 `(B,C,D,H,W)` 混用最容易错
- **视差维与空间维搞反**：`D` 是候选视差，不是高度
- **`disparity_regression` 输入必须是概率分布**：通常先过 softmax
- **拼接后通道数变化**：`concat=True` 会让通道翻倍，后续层要匹配
- **上采样不是简单插值**：`context_upsample` 依赖学习到的 `up_weights`

---

## 13. 推荐你的阅读顺序（30-45 分钟版）

1. 先读 `groupwise_correlation`（理解“像不像”怎么算）
2. 再读 `build_gwc_volume`（理解“按视差枚举”）
3. 看 `disparity_regression`（理解“概率 -> 连续视差”）
4. 看 `context_upsample`（理解“1/4 -> 全分辨率”）
5. 最后读 `CostVolumeDisparityAttention`（理解视差维全局建模）

如果你愿意，下一步我可以再给你做一份“带 shape 流水表”的版本：  
每个函数输入输出我用真实符号（比如 `B=2,C=128,D=48,H=96,W=160`）代入一遍，你会更快建立直觉。

