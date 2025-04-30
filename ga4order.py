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


def split_image(image: torch.Tensor, patch_size=16):
    """
    将 (C, H, W) 形状的图像分割成 14x14=196 个 16x16 小块，并编号
    - image: (C, 224, 224) 的张量
    - patch_size: 每个小块的大小，默认 16
    """
    image = image.squeeze(0)
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        "图像尺寸必须是 patch_size 的整数倍"
    )

    num_patches = (H // patch_size) * (W // patch_size)

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(
        num_patches, C, patch_size, patch_size
    )
    indices = np.arange(num_patches)
    return indices, patches


def rearrange_patches(patches, indices, new_order):
    """
    重新排列小块顺序
    - patches: (196, 3, 16, 16) 的张量
    - indices: (196,) 编号数组
    - new_order: (196,) 重新排列的索引顺序
    """
    assert len(new_order) == len(indices), "新顺序的大小必须等于196"
    reordered_patches = patches[new_order]
    return reordered_patches


def merge_patches(patches, indices, patch_size=16):
    """
    将 196 个小块合并成一张完整的图像
    - patches: (196, 3, 16, 16) 的张量
    - indices: (196,) 编号数组
    - patch_size: 每个小块的大小，默认 16
    """
    num_patches = len(indices)
    H, W = (
        patch_size * (int(np.sqrt(num_patches))),
        patch_size * (int(np.sqrt(num_patches))),
    )
    patches = patches.reshape(
        int(np.sqrt(num_patches)), int(np.sqrt(num_patches)), 3, patch_size, patch_size
    )
    patches = patches.permute(2, 0, 3, 1, 4).reshape(3, H, W)
    return patches


# def reorder(image: torch.Tensor, new_orders: np.array):
#     """
#     将图像重新排列
#     - image: (C, H, W) 的张量
#     - indices: (196,) 编号数组
#     - new_order: (196,) 重新排列的索引顺序
#     """
#     indices, patches = split_image(image)
#     rearrange_images = []
#     for new_order in new_orders:
#         if type(new_order) == np.float64:
#             new_order = new_orders
#         new_patches = patches.clone()
#         new_patches = rearrange_patches(new_patches, indices, new_order)
#         rearranged_image = merge_patches(new_patches, indices)
#         rearrange_images.append(rearranged_image)
#     rearranged_images = torch.stack(rearrange_images)
#     return rearranged_images

def reorder(image: torch.Tensor, new_order: np.array):
    """
    将图像重新排列
    - image: (C, H, W) 的张量
    - indices: (196,) 编号数组
    - new_order: (196,) 重新排列的索引顺序
    """
    indices, patches = split_image(image)
    patches = rearrange_patches(patches, indices, new_order)
    rearranged_image = merge_patches(patches, indices)
    return rearranged_image


def evaluate(outputs: torch.Tensor, targets: torch.Tensor):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型
    """
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses = criterion(outputs, targets)

    return losses


def evaluate_single(output: torch.Tensor, target: torch.Tensor):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)

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
        # output = torch.from_numpy(np.array(output))
    return outputs


def test_inf_single(model, image: torch.Tensor):
    """
    图像推理
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    """
    image = image.unsqueeze(0)  #B, C, H, W
    model.eval()
    output = model(image)
    # output = torch.from_numpy(np.array(output))
    return output


def f_ori(new_order, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """

    model, image, target = kwargs["model"], kwargs["image"], kwargs["target"]
    reordered_image = reorder(image, new_order)
    output = test_inf_single(model, reordered_image)
    loss = evaluate_single(output, target)

    # var = np.arange(196)
    # mismatch_num = np.sum(new_order != var)
    
    # return loss + 0.1 * mismatch_num
    return loss

def f(new_orders: np.array, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """
    model, image, target = kwargs["model"], kwargs["image"], kwargs["target"]

    reordered_images = []
    for new_order in new_orders:
        reordered_image = reorder(image, new_order)
        reordered_images.append(reordered_image)
    reordered_images = torch.stack(reordered_images)
    reordered_images = reordered_images.to(image.device)

    # reordered_images = reorder(image, new_orders)
    outputs = test_inf(model, reordered_images)
    targets = torch.full((reordered_images.size(0),), target.item()).to(image.device)
    losses = evaluate(outputs, targets)

    # var = np.arange(196)
    # mismatch_num = np.sum(new_order != var)

    # return loss + 0.1 * mismatch_num
    return losses


def save_images(new_order, image, original_image_path, reordered_image_path):
    """
    保存排列前后的图像
    - new_order: 重新排列的索引顺序
    - image: 原始图像
    - original_image_path: 原始图像保存路径
    - reordered_image_path: 重新排列图像保存路径
    """
    vutils.save_image(image, original_image_path)
    reordered_image = reorder(image, new_order)
    vutils.save_image(reordered_image, reordered_image_path)


def save_curve(total_fitness, curve_path):
    """
    保存收敛曲线
    - total_fitness: 每次搜索的适应度值
    - curve_path: 收敛曲线保存路径
    """

    max_length = max(len(fitness) for fitness in total_fitness)

    padded_fitness = []
    for fitness in total_fitness:
        if len(fitness) < max_length:
            padded = np.pad(
                fitness,
                (0, max_length - len(fitness)),
                mode="constant",
                constant_values=fitness[-1],
            )
        else:
            padded = np.array(fitness)
        padded_fitness.append(padded)

    padded_array = np.array(padded_fitness)
    avg_fitness = np.mean(padded_array, axis=0)

    plt.plot(avg_fitness)
    plt.xlabel("Iteration")
    plt.ylabel("Mean objective function")
    plt.title("Genetic Algorithm")
    plt.savefig(curve_path)


def solve(
    model,
    image: torch.Tensor,
    image_path: str,
    target: torch.Tensor,
    index: int,
    output: torch.Tensor,
    log_directory: str,
    search_times=10,
):
    """
    优化问题求解
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    - index: 图像索引
    """
    cnt = 0
    total_valid_solutions = []
    total_fitness = []
    while cnt < search_times:
        cnt += 1
        algorithm_param = {
            "max_num_iteration": 500,
            "population_size": 100,
            "mutation_probability": 0.2,
            "elit_ratio": 0.01,
            "crossover_probability": 0.8,
            "parents_portion": 0.3,
            "crossover_type": "pmx",
            "mutation_type": "swap",
            "init_type": "sequential",
            "max_iteration_without_improv": None,
            "improvement_threshold": None,
            "concurrent_processes": True,
        }

        m = ga(
            function=f if algorithm_param.get("concurrent_processes") else f_ori,
            dimension=196,
            function_timeout=10000,
            algorithm_parameters=algorithm_param,
            log_limit=100,
            log_directory=f"{log_directory}/{index}/{cnt}",
            model=model,
            image=image.detach(),
            target=target.detach(),
        )

        m.run()

        # 保存算法参数
        if not os.path.exists(f"{log_directory}/algorithm_parameter"):
            with open(f"{log_directory}/algorithm_parameter", "w") as file:
                for key, value in algorithm_param.items():
                    file.write(f"{key}: {value}\n")

        # 保存每次搜索结果
        dir_path = f"{log_directory}/{index}/{cnt}"
        os.makedirs(dir_path, exist_ok=True)
        np.savetxt(f"{dir_path}/logits.npy", output.cpu().detach().numpy().T)
        var = np.arange(196)
        mismatch_num = np.sum(m.best_variable.astype(int) != var)
        with open(f"{dir_path}/label_change.txt", "w") as file:
            file.write(
                f"{index}: \n\tgt_label: {target[0]} \n\tinf_label: {torch.argmax(output)} \n\tbeforeloss: {evaluate(output, target)} \n\tafter_loss: {m.best_function} \n\tmismatch_num: {mismatch_num} \n\timage_path: {image_path} \n\n"
            )
        original_image_path = f"{dir_path}/original_image.png"
        reordered_image_path = f"{dir_path}/reordered_image.png"
        save_images(m.best_variable, image, original_image_path, reordered_image_path)
        # 统计每次GA的loss和优解
        total_valid_solutions += m.output_dict["valid_solutions"]
        total_fitness.append(m.output_dict["fitness_per_inter"])

    np.save(
        f"{log_directory}/{index}/valid_solutions.npy",
        np.unique(np.array(total_valid_solutions), axis=0),
    )
    save_curve(total_fitness, f"{log_directory}/{index}/mean_convergence_curve.png")


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
        f"/home/wangsj2/Programs/order-matters/deit/wrong_classification/{current_time}"
    )
    for index in range(0, 1):
        image, target = dataset_val[index]
        image_path = dataset_val.imgs[index][0]
        image = image.to(device)
        image = image.unsqueeze(0)
        target = torch.tensor(target)
        target = target.to(device)
        target = target.unsqueeze(0)
        output = test_inf(model, image)
        solve(model, image, image_path, target, index, output, log_directory, 10)
        # if torch.argmax(output) != target:
        #     solve(model, image, image_path, target, index, output, log_directory, 10)
