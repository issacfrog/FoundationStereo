# FoundationStereo：零样本立体匹配

这是我们被 CVPR 2025 Oral（**Best Paper Nomination**）接收论文的官方实现。

[[项目主页]](https://nvlabs.github.io/FoundationStereo/) [[论文]](https://arxiv.org/abs/2501.09898) [[视频]](https://www.youtube.com/watch?v=R7RgHxEXB3o)

作者：Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

# 摘要
近年来，深度立体匹配在基准数据集上通过按域微调取得了巨大进展。然而，实现强零样本泛化（这是其他视觉任务中基础模型的典型特性）在立体匹配中仍然具有挑战性。我们提出 FoundationStereo：一个用于立体深度估计的基础模型，旨在实现强零样本泛化。为此，我们首先构建了一个大规模（100 万双目图像对）的合成立体训练数据集，具有高多样性和高写实性，并通过自动自筛选流程移除模糊样本。随后我们设计了一系列网络结构组件以增强可扩展性，包括一种侧向微调特征骨干网络，可利用视觉基础模型中的丰富单目先验来缓解仿真到真实域差距，以及长程上下文推理模块，用于更有效的代价体过滤。上述设计共同带来了跨域场景下更强的鲁棒性与精度，为零样本立体深度估计建立了新标准。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/intro.jpg" width="800"/>
</p>

**TLDR**：我们的方法输入一对双目图像，输出一张稠密视差图；该视差图可进一步转换为米制深度图或 3D 点云。

<p align="center">
  <img src="./teaser/input_output.gif" width="600"/>
</p>

# 更新日志
| 日期       | 说明                                                                                                                |
|------------|---------------------------------------------------------------------------------------------------------------------|
| 2025/12/15 | 发布实时模型 [Fast-FoundationStereo](https://nvlabs.github.io/Fast-FoundationStereo/)                              |
| 2025/08/05 | 商业模型已上线，见 [这里](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)!           |
| 2025/07/03 | 改进 ONNX 和 TRT 支持，并新增 Jetson 支持                                                                          |

# 榜单成绩
我们在 [Middlebury 榜单](https://vision.middlebury.edu/stereo/eval3/) 和 [ETH3D 榜单](https://www.eth3d.net/low_res_two_view) 上获得世界第一。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/middlebury_leaderboard.jpg" width="700"/>
  <br>
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/eth_leaderboard.png" width="700"/>
</p>

# 与单目深度估计对比
我们的方法在不同场景的零样本立体匹配任务上优于现有方法。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/mono_comparison.png" width="700"/>
</p>

# 安装

我们已在 Linux + GPU（3090、4090、A100、V100、Jetson Orin）上测试通过。其他 GPU 理论上也可运行，但请确保显存充足。

```
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo
```

注意：`flash-attn` 需要单独安装，以避免[环境创建时报错](https://github.com/NVlabs/FoundationStereo/issues/20)。

# 模型权重
- 下载用于零样本推理的基础模型权重，并将整个文件夹（例如 `23-51-11`）放到 `./pretrained_models/` 下。

| 模型     | 说明                                                                                       |
|-----------|--------------------------------------------------------------------------------------------|
| [23-51-11](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | 通用场景下表现最好的模型，基于 ViT-Large |
| [11-33-40](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | 精度略低但推理更快，基于 ViT-Small |
| [NVIDIA-TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)       | 商业版本（基于 ViT-Small 适配） |

# 运行 Demo
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```
你将看到输出点云。

<p align="center">
  <img src="./teaser/output.jpg" width="700"/>
</p>

提示：
- 输入的左右图应当是**已校正且去畸变**的，即不存在鱼眼畸变，且左右图间极线水平。如果你使用 ZED 等双目相机，通常已完成此步骤（参考[这里](https://github.com/stereolabs/zed-sdk/blob/3472a79fc635a9cee048e9c3e960cc48348415f0/recording/export/svo/python/svo_export.py#L124)）。
- 不要交换左右图。左图必须来自左相机（物体在左图中应更靠右显示）。
- 建议使用无损压缩的 PNG 图像。
- 方法对 RGB 双目图像效果最佳；但我们也测试过单色或红外双目图像（如 RealSense D4XX 系列），同样可用。
- 所有参数请通过 `python scripts/run_demo.py --help` 查看。
- 若要为你自己的数据生成点云，需要提供内参文件：第一行为展平的 `1x9` 相机内参矩阵，第二行为左右相机基线（单位：米）。
- 对高分辨率图像（>1000 像素），可选择：  
  (1) 使用 `--hiera 1` 开启分层推理，得到全分辨率深度但更慢；  
  (2) 使用更小缩放，如 `--scale 0.5`，得到降采样分辨率深度但更快。
- 若要更快推理，可降低输入分辨率（如 `--scale 0.5`）并减少迭代次数（如 `--valid_iters 16`）。

# ONNX / TensorRT (TRT) 推理

我们目前仅支持通过 Docker 进行 ONNX/TRT 推理。

- 构建 Docker（测试环境：NVIDIA Driver 560.35.03，CUDA 12.6）
```bash
export DIR=$(pwd)
cd docker && docker build --network host -t foundation_stereo .
bash run_container.sh
cd /
git clone https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt
python3 setup.py install
apt-get install -y libnvinfer-dispatch10 libnvinfer-bin tensorrt
cd $DIR
```

- 导出 ONNX：
```
XFORMERS_DISABLED=1 python scripts/make_onnx.py --save_path ./pretrained_models/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 448 --width 672 --valid_iters 20
```

- 转换为 TRT：
```
trtexec --onnx=pretrained_models/foundation_stereo.onnx --verbose --saveEngine=pretrained_models/foundation_stereo.plan --fp16
```

- 运行 TRT：
```
python scripts/run_demo_tensorrt.py \
        --left_img ${PWD}/assets/left.png \
        --right_img ${PWD}/assets/right.png \
        --save_path ${PWD}/output \
        --pretrained pretrained_models/foundation_stereo.plan \
        --height 448 \
        --width 672 \
        --pc \
        --z_far 100.0
```

我们在同一张 3090 GPU 上观察到 TensorRT FP16 可达到约 6 倍加速。实际加速比取决于多种因素，如果你关心速度，建议尝试该路径，并根据需求调整参数。

# 在 Jetson 上运行
请参考 [readme_jetson.md](readme_jetson.md)。

# FSD 数据集
<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/sdg_montage.jpg" width="800"/>
</p>

你可以在[这里](https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing)下载完整数据集（>1TB）。我们也提供了一个用于快速浏览的小样本数据（3GB），见[这里](https://drive.google.com/file/d/1dJwK5x8xsaCazz5xPGJ2OKFIWrd9rQT5/view?usp=drive_link)。完整数据集约包含 100 万条数据，每条包括：
- 左右图像
- 真实视差（ground-truth disparity）

你可以通过示例读取小样本数据：
```
python scripts/vis_dataset.py --dataset_path ./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/
```

将得到：
<p align="center">
  <img src="./teaser/fsd_sample.png" width="800"/>
</p>

关于数据集许可，请查看[这里](https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE)。

# FAQ
- 问：Conda 安装失败怎么办？  
  答：参考[这个 issue](https://github.com/NVlabs/FoundationStereo/issues/20)。

- 问：没有得到点云，或点云不完整怎么办？  
  答：检查 argparse 中与点云处理相关参数，例如 `--z_far`、`--remove_invisible`、`--denoise_cloud`。

- 问：我的 GPU 不支持 Flash Attention 怎么办？  
  答：参考[这个回复](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825)。

- 问：报错 `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED...` 怎么办？  
  答：这通常表示显存不足（OOM）导致的问题。请降低输入分辨率或使用显存更大的 GPU。

- 问：如何在 RealSense 上运行？  
  答：参考[这里](https://github.com/NVlabs/FoundationStereo/issues/26)和[这里](https://github.com/NVlabs/FoundationStereo/issues/80)。

- 问：我有两台或多台 RGB 相机，可以使用吗？  
  答：可以先用这个 [OpenCV 函数](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6) 将一对图像校正为标准双目图像对（相对旋转被消除），再输入 FoundationStereo。

- 问：如何在 Windows 上运行？  
  答：参考[这里](https://github.com/NVlabs/FoundationStereo/issues/219)。

- 问：可以用于商业用途吗？  
  答：我们已发布商业版本，见[这里](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)。也可邮件联系 bowenw@nvidia.com。

# 引用
```
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```

# 致谢
感谢 Gordon Grigor、Jack Zhang、Karsten Patzwaldt、Hammad Mazhar 以及其他 NVIDIA Isaac 团队成员的大力工程支持与宝贵讨论。感谢 [DINOv2](https://github.com/facebookresearch/dinov2)、[DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[Selective-IGEV](https://github.com/Windsrain/Selective-Stereo) 和 [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) 的作者开源代码。也感谢 CVPR 审稿人与 AC 的认可和建设性反馈。

# 联系方式
如有商业合作、技术支持或其他问题，请联系 [Bowen Wen](https://wenbowen123.github.io/)（bowenw@nvidia.com）。
