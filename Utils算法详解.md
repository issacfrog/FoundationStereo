# FoundationStereo `Utils.py` 算法详解（零基础版）

这份文档专门讲 `Utils.py`，适合你现在这种“边看代码边学基础”的节奏。  
这个文件不属于主网络结构，而是一些高频通用工具：日志、随机种子、点云、深度几何、模型冻结、尺寸对齐、视差可视化、深度解码。

---

## 1. 文件整体作用

`Utils.py` 主要做三类事：

- **工程基础工具**：日志格式、随机种子、冻结模型参数
- **几何与数据处理**：深度图转 3D 点、uint8 深度解码
- **显示与输入适配**：视差可视化、图像尺寸对齐

简单说：它是“算法主干之外的支撑层”。

---

## 2. 逐函数详解

---

### 2.1 `set_logging_format(level=logging.INFO)`

#### 作用

统一日志输出格式，便于控制台阅读。

#### 关键逻辑

- `importlib.reload(logging)`：重载 logging，避免旧配置残留
- `FORMAT = '%(message)s'`：只显示日志正文，不显示冗余前缀
- `logging.basicConfig(...)`：设置等级和格式

#### 参数

- `level`：日志等级，默认 `INFO`

#### 备注

文件里 `set_logging_format()` 在定义后立刻调用，所以默认日志格式会在导入本文件时生效。

---

### 2.2 `set_seed(random_seed)`

#### 作用

设置随机种子，提升结果可复现性（训练/推理调试都常用）。

#### 做了什么

- 设定 `numpy` 随机种子
- 设定 Python `random` 随机种子
- 设定 `torch` CPU/GPU 随机种子
- `cudnn.deterministic=True`：尽量确定性
- `cudnn.benchmark=False`：关闭自动算法搜索，减少非确定性

#### 参数

- `random_seed`：整型随机种子，例如 `42`

#### 注意

“尽量复现”不等于“100% 位级一致”，但通常会显著稳定实验结果。

---

### 2.3 `toOpen3dCloud(points, colors=None, normals=None)`

#### 作用

把 `numpy` 点云数据转成 Open3D 的 `PointCloud` 对象，方便可视化/保存。

#### 参数

- `points`：`(N,3)`，每行一个 3D 点 `(x,y,z)`
- `colors`（可选）：`(N,3)`，RGB；如果值域大于 1，函数会自动除以 255
- `normals`（可选）：`(N,3)`，法向量

#### 返回

- `o3d.geometry.PointCloud` 对象

#### 典型用途

把深度重建出的点云直接丢给 Open3D 显示。

---

### 2.4 `depth2xyzmap(depth, K, uvs=None, zmin=0.1)`

#### 作用（几何核心）

把深度图转换为三维坐标图 `xyz_map`。

#### 输入参数

- `depth`：深度图，形状 `(H,W)`，单位通常是米
- `K`：相机内参矩阵（`3x3`）
  - `K[0,0]=fx`, `K[1,1]=fy`
  - `K[0,2]=cx`, `K[1,2]=cy`
- `uvs`（可选）：指定只计算某些像素点，形状 `(N,2)`
- `zmin`：最小有效深度阈值，小于这个值视为无效

#### 数学关系

对每个像素 `(u,v)` 及深度 `z`：

- `x = (u - cx) * z / fx`
- `y = (v - cy) * z / fy`
- `z = depth(v,u)`

#### 输出

- `xyz_map`：形状 `(H,W,3)`，每个像素位置对应一个 3D 点
- 无效深度位置会被置为 `0`

#### 直觉

它做的是“把 2D 深度图反投影到相机坐标系 3D 空间”。

---

### 2.5 `freeze_model(model)`

#### 作用

冻结模型参数，常用于“把预训练模型当特征提取器”。

#### 做了什么

- `model.eval()`：切到推理模式
- 对所有 `parameters()`：`requires_grad=False`
- 对所有 `buffers()`：也设置 `requires_grad=False`（缓冲区一般不训练）

#### 返回

- 冻结后的同一个 `model`

#### 在本项目中的典型场景

`extractor.py` 里会冻结 `DepthAnything`，只把它当固定特征源。

---

### 2.6 `get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232)`

#### 作用（很常用）

根据原图尺寸计算一个“合法输入尺寸”：

- 尽量保持长宽比
- 高宽都对齐到 `divider` 的倍数
- 不超过 `max_H/max_W`

#### 参数

- `H, W`：原图高宽
- `divider`：对齐倍数（例如 16、224 等）
- `max_H, max_W`：尺寸上限

#### 逻辑步骤

1. 先把 `H/W` 向上取整到 `divider` 倍数
2. 若超过上限，按长边约束等比缩小
3. 缩小后再次做 `divider` 对齐
4. 返回新尺寸 `(H_resize, W_resize)`

#### 注意点

- 这个函数是“重采样尺寸计算”，不直接改图像；真正 resize 在别处（如 `F.interpolate`）
- 返回结果可能比原图大也可能小，取决于输入与约束

---

### 2.7 `vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={})`

#### 作用

把视差图转换成彩色可视化图，便于人眼观察。

#### 参数

- `disp`：视差图 `(H,W)`
- `min_val/max_val`：可视化区间；若不传，自动从有效像素统计
- `invalid_thres`：大于此阈值视为无效
- `color_map`：OpenCV 颜色映射（默认 `TURBO`）
- `cmap`：可选自定义 colormap 函数
- `other_output`：字典，函数会写回 `min_val/max_val`

#### 处理流程

1. 复制 `disp` 避免原地修改
2. 计算无效掩码
3. 归一化到 `0~255`
4. 应用 colormap 得到彩色图
5. 无效区域置黑

#### 返回

- `uint8` 彩色图 `(H,W,3)`

#### 小提醒

函数签名里的 `other_output={}` 是可变默认参数，工程里一般建议避免；但当前用法问题不大。

---

### 2.8 `depth_uint8_decoding(depth_uint8, scale=1000)`

#### 作用

把 3 通道 `uint8` 编码深度图解码回浮点深度值。

#### 输入

- `depth_uint8`：`(H,W,3)`，每个像素 3 个字节表示一个深度整数
- `scale`：缩放因子（默认 `1000`）

#### 解码公式

- `raw = c0*255*255 + c1*255 + c2`
- `depth = raw / scale`

#### 返回

- 浮点深度图 `(H,W)`

#### 直觉

这是把“分成三个 8-bit 通道存储的深度”还原成真实数值。

---

## 3. 这几个函数怎么串起来用（实战视角）

常见一条小流水线：

1. `set_seed` 固定随机性
2. 读取模型并 `freeze_model`（若只做特征提取）
3. 用 `get_resize_keep_aspect_ratio` 算输入尺寸并 resize 图像
4. 模型输出视差后，用 `vis_disparity` 做可视化
5. 若有深度图，`depth2xyzmap` 转 3D，再 `toOpen3dCloud` 可视化点云

---

## 4. 初学者最容易混淆的点

- `get_resize_keep_aspect_ratio` 只算尺寸，不改图像本身
- `depth2xyzmap` 需要相机内参 `K`，不是只靠深度图就够
- `freeze_model` 后模型不会更新参数，适合“固定特征提取”
- `vis_disparity` 的颜色只用于显示，不是数值本身
- `depth_uint8_decoding` 的 `scale` 要与编码时一致，否则深度会错量级

---

## 5. 你可以立刻做的 10 分钟小练习

1. 选一张视差图 `disp`，跑 `vis_disparity` 看彩色结果  
2. 打印 `other_output` 里的 `min_val/max_val`，理解颜色映射区间  
3. 用一张深度图 + 相机内参跑 `depth2xyzmap`，检查输出形状是不是 `(H,W,3)`  
4. 随机抽几个像素，验证 `x,y,z` 变化是否符合直觉（中心点接近光轴）

这样你会很快把这些工具函数从“看懂代码”变成“会用”。

