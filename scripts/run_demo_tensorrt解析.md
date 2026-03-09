# `scripts/run_demo_tensorrt.py` 代码解析（ONNX/Engine 推理版）

这个脚本做的是：**用 ONNXRuntime 或 TensorRT Engine 跑单对左右图推理**，并输出可视化图/可选点云。

---

## 1. 功能概览

它支持两种模型格式：

- `.onnx`：用 ONNXRuntime（CUDA provider）
- `.engine` / `.plan`：用 TensorRT engine（反序列化后执行）

输出内容：

- 视差可视化拼图（左图 + 伪彩视差）
- 可选 `.ply` 点云（`--pc`）

---

## 2. 关键函数拆解

## `preprocess(image_path, args)`

作用：读图并转成模型输入 tensor。

- `imageio.imread` 读图
- 若指定 `--height --width` 则 `cv2.resize`
- 转 torch tensor，形状变成 `(1,3,H,W)`
- 返回：
  - `resized_image`：模型输入
  - `input_image`：用于可视化拼接的原图（或 resize 后图）

---

## `get_onnx_model(args)`

作用：创建 ONNXRuntime Session。

- 打开图优化 `ORT_ENABLE_ALL`
- provider 设为 `CUDAExecutionProvider`

---

## `get_engine_model(args)`

作用：读取 TensorRT engine 二进制并反序列化。

- `trt.Runtime(...).deserialize_cuda_engine(engine_data)`
- 再包装成 `onnx_tensorrt.tensorrt_engine.Engine`

---

## `inference(left_img_path, right_img_path, model, args)`

这是核心推理流程：

1. 左右图预处理
2. 循环 10 次做计时（便于观察延迟）
3. 根据模型格式走不同推理接口：
   - ONNX: `model.run(None, {'left':..., 'right':...})`
   - TensorRT Engine: `model.run([left, right])`
4. `left_disp.squeeze()` 得到 `HxW`
5. `vis_disparity(left_disp)` 生成伪彩视差图
6. 与左图横向拼接后保存到 `visual/`
7. 若 `--pc`，用固定 `K`、`baseline` 做深度恢复并写 `.ply`

---

## 3. 参数说明（`parse_args`）

- `--left_img / --right_img`：输入左右图路径（必填）
- `--save_path`：输出目录
- `--pretrained`：模型文件路径（`.onnx` 或 `.engine/.plan`）
- `--height --width`：推理输入尺寸
- `--pc`：是否保存点云
- `--z_far`：点云保留的最大深度阈值

---

## 4. `main()` 流程

1. 解析参数
2. 创建输出目录：
   - `continuous/disparity`
   - `visual`
   - `denoised_cloud`
   - `cloud`
3. 检查模型文件存在
4. `set_seed(0)`
5. 按后缀选择 ONNX 或 Engine 加载
6. 调用 `inference(...)`

---

## 5. 结果与几何关系

脚本里点云深度公式：

\[
depth = \frac{f_x \cdot baseline}{disp + doffs}
\]

其中：

- `f_x = K[0,0]`
- `baseline` 是双目基线
- `doffs` 这里写死为 0

再通过 `depth2xyzmap` 把深度转成 3D 点，最后用 Open3D 输出 PLY。

---

## 6. 注意事项

- 输入 tensor 没有显式归一化；需确认导出模型是否内置预处理
- 这里的 `K` 与 `baseline` 是固定常量，换数据集要替换
- 计时循环重复 10 次，日志里是每次耗时，不是平均值
- ONNX 与 TensorRT 的输入名称/接口不同，脚本已分支处理

---

## 7. 一句话总结

`run_demo_tensorrt.py` 是一个部署验证脚本：  
**加载 ONNX/Engine，跑单对图推理，输出可视化并可选导出点云。**

