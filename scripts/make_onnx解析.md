# `scripts/make_onnx.py` 代码解析（零基础版）

这个脚本的目标很单一：**把 FoundationStereo 的 PyTorch 权重导出成 ONNX 模型**，用于部署。

---

## 1. 脚本做了什么（先看全局）

主流程：

1. 解析命令行参数（导出路径、checkpoint、输入尺寸、迭代次数）
2. 读取 checkpoint 同目录下的 `cfg.yaml`
3. 构建一个“ONNX 导出友好”的模型包装类 `FoundationStereoOnnx`
4. 加载权重并切到 `eval + cuda`
5. 构造随机输入张量
6. 调用 `torch.onnx.export(...)` 导出 `.onnx`

---

## 2. 关键类：`FoundationStereoOnnx`

```python
class FoundationStereoOnnx(FoundationStereo):
```

它继承原模型，只改了 `forward(left, right)` 的行为，让导出更稳定：

- 固定走 `test_mode=True`（只输出最终视差）
- 固定 `iters=self.args.valid_iters`
- 用 `torch.amp.autocast('cuda', enabled=True)` 包裹

这相当于把训练时的“多输出/复杂分支”简化成部署时真正需要的单输出 `disp`。

---

## 3. 参数说明（`argparse`）

- `--save_path`：ONNX 输出路径
- `--ckpt_dir`：PyTorch 预训练权重路径（`.pth`）
- `--height/--width`：导出时 dummy 输入尺寸
- `--valid_iters`：模型内部迭代细化次数

---

## 4. 配置与权重加载逻辑

核心片段：

1. `cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')`
2. 用命令行参数覆盖 `cfg`
3. 若没有 `vit_size`，补默认 `vitl`
4. `model = FoundationStereoOnnx(cfg)`
5. `ckpt = torch.load(ckpt_dir)`，再 `model.load_state_dict(ckpt['model'])`

为什么这样做：  
训练时配置和权重是绑定的，导出也要尽量复现训练配置。

---

## 5. ONNX 导出细节

导出调用：

- `opset_version=16`
- 输入名：`left`, `right`
- 输出名：`disp`
- 动态轴：仅 batch 维动态

```python
dynamic_axes={
  'left': {0: 'batch_size'},
  'right': {0: 'batch_size'},
  'disp': {0: 'batch_size'}
}
```

含义：导出的模型支持不同 batch size，但高宽在这份导出里是固定的（由 dummy 输入决定）。

---

## 6. 输入输出形状直觉

导出时输入：

- `left_img`: `(1,3,height,width)`
- `right_img`: `(1,3,height,width)`

输出：

- `disp`: 通常是 `(1,1,height,width)`（具体依模型 forward）

---

## 7. 使用示例

```bash
python scripts/make_onnx.py \
  --ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth \
  --save_path output/foundation_stereo.onnx \
  --height 448 --width 672 --valid_iters 16
```

---

## 8. 常见问题与排查

- **导出报 CUDA 相关错误**：确认在 GPU 环境运行，且权重和模型都 `.cuda()`
- **导出后推理尺寸不匹配**：当前只开了 batch 动态，若要动态高宽需额外配置 dynamic axes
- **权重加载失败（key 不匹配）**：检查 `cfg.yaml` 与 `ckpt` 是否配套
- **输出结果和训练推理不一致**：确认 `valid_iters` 与推理脚本一致

---

## 9. 一句话总结

`make_onnx.py` 是“**从训练权重到部署模型**”的转换桥梁：  
读取配置和权重，包装成部署路径，导出单输出 ONNX。

