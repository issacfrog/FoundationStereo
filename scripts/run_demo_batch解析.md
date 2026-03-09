# `scripts/run_demo_batch.py` 代码解析（批量/多相机推理版）

这个脚本是三者里最“工程化”的一个：  
它读取 metadata，自动匹配左右相机帧，批量跑 FoundationStereo，输出深度图（可选点云），并支持多 GPU 分工。

---

## 1. 总体目标

输入：

- 图像目录 `--imgdir`
- 元数据 `--metadata_file`
- 预训练权重 `--ckpt_dir`

输出：

- `scaled_0_4/`：按推理缩放尺度下的 16bit 深度图
- `original_size/`：恢复到原分辨率的 16bit 深度图
- 可选 `pcd/`：点云文件

---

## 2. 主类：`CuSFMDataInference`

这个类把流程拆成“加载元数据、加载模型、逐相机处理”。

## `__init__`

- 绑定 `args`
- 选设备（`cuda:{rank}`）
- 调 `load_metadata()` + `load_model()`

---

## `load_metadata`

读 `metadata_file`（json），解析出：

- `camera_params_id_to_camera_params`
- `stereo_pair`（左右相机对应关系、baseline）
- `keyframes_metadata`

并整理成：

- `self.camera_keyframes[camera_param_id][synced_sample_id] = keyframe`

作用：后续可按 `synced_sample_id` 快速找到左右同帧图像。

---

## `load_model`

1. 读取 checkpoint 同目录的 `cfg.yaml`
2. `vit_size` 不存在则补默认 `vitl`
3. 构建 `FoundationStereo(cfg)`
4. 加载 `ckpt['model']`
5. 放到当前设备并 `eval`

---

## `get_projection_matrix` / `get_camera_transform`

这两个函数用于几何参数：

- 投影矩阵 `P(3x4)` -> 取前 `3x3` 作为 `K`
- 传感器到车体变换（平移 + 轴角旋转）

---

## `compute_baseline`

当 metadata 未直接给 baseline 时，代码用左右相机外参推基线：

1. 构建左右相机到车体的齐次变换矩阵
2. 计算左到右变换
3. 从平移项提取 baseline（x 方向）

若算出来为 0 会报错（深度无法计算）。

---

## `process_camera(left_param_id)`（核心）

这是整份脚本最重要函数，流程如下：

1. 取左相机参数、相机名、内参 `K`
2. 在 stereo_pair 里找对应右相机 id 与 baseline
3. 拿到左右 keyframe 字典（按 `synced_sample_id` 对齐）
4. 遍历左帧，找同 `synced_sample_id` 的右帧
5. 读左右图，按 `scale` 缩放（默认 0.4）
6. 转 tensor `(1,3,H,W)`，并用 `InputPadder` pad 到 32 对齐
7. 混合精度下推理：
   - `disp = model.forward(..., test_mode=True)`
   - 再 `unpad`
8. `disp` 转 numpy，并 resize 回原分辨率得到 `disp_big`
9. 按公式转深度（毫米）：

\[
depth = 1000 \cdot scale \cdot f_x \cdot baseline / (disp + doffs)
\]

10. 截断超大深度（>65535 置 0），保存成 `uint16 png`
11. 若开启 `also_generate_for_right_camera`，对右相机也生成深度
12. 若 `get_pc=True`，再把深度转点云并保存 `.ply`

---

## 3. 右相机生成逻辑（可选）

当 `--also_generate_for_right_camera=True`：

- 先把输入水平翻转
- 用“右->左”顺序跑一次模型
- 输出再翻转回来

这是常见的“同模型近似生成另一视角深度”的工程技巧。

---

## 4. 多 GPU 分发逻辑

在 `main(rank, world_size, args)`：

1. 筛选需要处理的左相机列表（支持 `--camera` 或 `--cameras`）
2. 仅保留左相机（`is_left_camera`）
3. 按 GPU 数量均分相机 ID
4. 每个 rank 只处理自己分到的相机

入口：

- `num_gpus > 1`：`mp.spawn(main, ...)`
- 否则单进程执行

---

## 5. 参数说明（高频）

- `--imgdir`：图像根目录（必填）
- `--metadata_file`：元数据 json（必填）
- `--out_dir`：输出目录（必填）
- `--ckpt_dir`：权重文件
- `--scale`：推理缩放比例（默认 0.4）
- `--valid_iters`：迭代细化轮数
- `--get_pc`：是否保存点云
- `--z_far`：点云深度裁剪上限
- `--num_gpus`：使用 GPU 数量
- `--also_generate_for_right_camera`：是否也生成右相机深度

---

## 6. 代码里的工程特点与注意点

- `InputPadder(divis_by=32)`：保证尺寸可被网络下采样路径整除
- 输出深度用 `uint16 png`：便于存储，范围受 65535 限制
- `type=bool` 的 argparse 参数在命令行上不太稳，实际使用建议改成 `action='store_true'`
- 点云生成前会按 `z_far` 过滤远点，避免无意义噪点
- metadata 若缺部分字段，代码里做了默认兼容（如 `camera_params_id='0'`）

---

## 7. 一句话总结

`run_demo_batch.py` 是“**面向真实数据集批处理**”的完整推理管线：  
自动按 metadata 配对左右图，批量出深度（和可选点云），还能按多 GPU 分摊工作量。

