# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # 参数解析
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)                              # 设置随机种子 设置为固定的0之后避免了一些随机的问题
  torch.autograd.set_grad_enabled(False)   # 全局关闭梯度，表明不是训练
  os.makedirs(args.out_dir, exist_ok=True) # 创建输出目录

  ckpt_dir = args.ckpt_dir
  # Load training-time config stored next to checkpoint, then override with CLI args.
  # 加载pth训练参数
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args) # 根据加载的参数创建 FoundationStereo

  # [Modified by Assistant] PyTorch>=2.6 defaults torch.load(weights_only=True),
  # but this checkpoint stores extra Python objects (e.g., numpy scalar metadata).
  # Load with weights_only=False for trusted local checkpoints.
  ckpt = torch.load(ckpt_dir, map_location='cpu', weights_only=False) # 先在cpu端对读取的checkpoint文件进行反序列化
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model']) # 将模型参数加载到model中

  model.cuda() # 将模型移动到cuda设备上
  model.eval() # 设置模型为评估模式

  code_dir = os.path.dirname(os.path.realpath(__file__))
  # Read rectified stereo pair from disk.
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  scale = args.scale
  assert scale<=1, "scale must be <=1"
  # Optional input downscale for speed/memory trade-off.
  # 可选对输入图像进行下采样
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None) # 这里返回的是numpy格式的
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2] # 记录缩放后的尺寸
  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  # as_tensor：numpy 转 torch
  # .cuda()：放到 GPU
  # .float()：转 float32
  # [None]：加 batch 维，HWC -> 1HWC
  # .permute(0,3,1,2)：改成模型常用格式 NCHW
  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2) # permute函数调整的是当前张量维度的顺序
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  # Model expects spatial sizes divisible by 32.
  padder = InputPadder(img0.shape, divis_by=32, force_square=False) # 对齐到网络友好的张量尺寸
  img0, img1 = padder.pad(img0, img1) # 执行对齐操作

  # Stereo inference: either single-scale forward or hierarchical forward for large images.
  with torch.cuda.amp.autocast(True): # 开启混合精度
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True) # 单尺度推理
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5) # 分层推理
  # 推理的这些细节后面再读

  # 推理得到的结果是双目视差图
  disp = padder.unpad(disp.float()) # unpad：移除填充，恢复原始尺寸
  disp = disp.data.cpu().numpy().reshape(H,W) # 将disp张量转换为numpy数组，并重塑为(H,W)形状 

  # Save side-by-side visualization (left RGB + colorized disparity).
  # 将推理得到的视差图进行可视化 等操作
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/vis.png', vis)
  logging.info(f"Output saved to {args.out_dir}")

  # 去除掉非共视区域
  if args.remove_invisible:
    # Remove pixels that map outside right image after disparity shift.
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    # Convert disparity to metric depth using intrinsics and baseline.
    # 深度转换操作
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    # Keep only forward-facing points within z_far.
    # 远处点的过滤
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")

    # 去噪&可视化
    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()

