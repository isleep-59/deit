import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import matplotlib.pyplot as plt

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset

# from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils
import torchvision.utils as vutils
from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga
import datetime
import os
from PIL import Image

from typing import Tuple


def split_image(images: torch.Tensor, patch_size=16):
    """
    将 (B, C, H, W) 形状的图像分割成 14x14=196 个 16x16 小块，并编号
    - image: (C, 224, 224) 的张量
    - patch_size: 每个小块的大小，默认 16
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        "图像尺寸必须是 patch_size 的整数倍"
    )

    num_patches = (H // patch_size) * (W // patch_size)

    patches = images.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(B, num_patches, C, patch_size, patch_size)

    return patches


def rearrange_patches(patches: torch.Tensor, new_order: torch.Tensor):
    """
    根据提供的顺序重新排列批量图像的小块。
    假设对批次中的每个图像应用相同的重新排列顺序。

    - patches: (B, num_patches, C, patch_size, patch_size) 的张量
    - new_order: (num_patches,) 的张量或列表/数组，包含 0 到 num_patches-1 的索引，
                 指定了小块的新顺序。

    返回:
    - reordered_patches: (B, num_patches, C, patch_size, patch_size) 的张量，
                         其中每个图像的小块已按 new_order 重新排列。
    """
    B, num_patches = patches.shape[0], patches.shape[1]

    if not isinstance(new_order, torch.Tensor):
        new_order = torch.tensor(new_order, dtype=torch.long, device=patches.device)
    elif new_order.device != patches.device:
        new_order = new_order.to(patches.device)
    if new_order.dim() != 1:
        raise ValueError(f"new_order 必须是一维张量，但得到了 {new_order.dim()} 维")

    assert new_order.shape[0] == num_patches, \
        f"新顺序 new_order 的长度 ({new_order.shape[0]}) 必须等于小块数量 ({num_patches})"

    reordered_patches = patches[:, new_order]

    return reordered_patches


def merge_patches(patches: torch.Tensor, output_size: Tuple[int, int]):
    """
    将批量的小块合并回完整的批量图像。
    假设输入的 patches 是按照先行后列 (raster scan) 的原始顺序排列的。

    - patches: (B, num_patches, C, patch_size, patch_size) 的张量
    - output_size: 目标输出图像的尺寸 (H, W) 元组

    return:
    - images: (B, C, H, W) 的张量
    """
    B, num_patches_in, C, ps_h, ps_w = patches.shape
    H, W = output_size

    if ps_h != ps_w:
        raise ValueError(f"当前实现假定 patch 是方形的, 但得到 H={ps_h}, W={ps_w}")
    patch_size = ps_h

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"目标图像尺寸 H={H}, W={W} 必须能被 patch_size={patch_size} 整除")

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    expected_num_patches = num_patches_h * num_patches_w

    if num_patches_in != expected_num_patches:
        raise ValueError(f"输入的 patch 数量 ({num_patches_in}) 与基于 output_size "
                         f"({H}x{W}) 和 patch_size ({patch_size}) 计算的数量 "
                         f"({expected_num_patches}) 不匹配")

    patches_grid = patches.reshape(B, num_patches_h, num_patches_w, C, patch_size, patch_size)

    patches_permuted = patches_grid.permute(0, 3, 1, 4, 2, 5)

    images = patches_permuted.reshape(B, C, H, W)

    return images


def reorder(images: torch.Tensor, new_order: torch.Tensor, patch_size: int = 16):
    """
    将图像重新排列
    - image: (B, C, H, W) 的张量
    - new_order: (196,) 重新排列的索引顺序
    """
    B, C, H, W = images.shape
    expected_num_patches = (H // patch_size) * (W // patch_size)
    if H % patch_size != 0 or W % patch_size != 0:
         raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})")
    if len(new_order) != expected_num_patches:
        raise ValueError(f"Length of new_order ({len(new_order)}) does not match "
                         f"expected number of patches ({expected_num_patches}) "
                         f"for image size {H}x{W} and patch_size {patch_size}")
    
    patches = split_image(images, patch_size=patch_size)
    patches = rearrange_patches(patches, new_order)
    rearranged_images = merge_patches(patches, output_size=(H, W))
    return rearranged_images


def evaluate(outputs: torch.Tensor, targets: torch.Tensor):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型

    return: 每个batch的平均损失值
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    return loss


def test_inf(model, images: torch.Tensor):
    """
    图像推理
    - model: 模型
    - image: (B, C, H, W) 的图像张量
    - target: 目标标签
    """
    with torch.no_grad():
        model.eval()
        outputs = model(images)
    return outputs


def f(new_order: np.array, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """
    model, data_loader, device = kwargs["model"], kwargs["data_loader"], kwargs["device"]
    losses = []
    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        reordered_images = reorder(images, new_order, patch_size=16)
        outputs = test_inf(model, reordered_images)
        batch_loss = evaluate(outputs, targets)
        losses.append(batch_loss)

    # 计算平均损失
    avg_loss = sum(losses) / len(losses)
    return avg_loss.item()


def solve(
    model,
    data_loader_val: torch.utils.data.DataLoader,
    log_directory: str,
):
    """
    优化问题求解
    - model: 模型
    - data_loader_val: 验证数据加载器
    - log_directory: 日志目录
    """
    
    algorithm_param = {
        "max_num_iteration": 1000,
        "population_size": 50,
        "mutation_probability": 0.8,
        "elit_ratio": 0.02,
        "crossover_probability": 0.8,
        "parents_portion": 0.3,
        "crossover_type": "pmx",
        "mutation_type": "swap",
        "init_type": "random",
        "max_iteration_without_improv": None,
        "improvement_threshold": None,
        "concurrent_processes": False,
    }

    m = ga(
        function=f,
        dimension=196,
        function_timeout=10000,
        algorithm_parameters=algorithm_param,
        log_limit=100,
        log_directory=f"{log_directory}",
        model=model,
        data_loader=data_loader_val,
        device="cuda",
    )

    m.run()


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )

    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_directory = (
        f"/home/wangsj2/Programs/order-matters/deit/universal_order/{current_time}"
    )

    solve(model, data_loader_val, log_directory)
