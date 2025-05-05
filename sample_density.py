
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

import argparse
from pathlib import Path
import sys

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
import random
from PIL import Image

def split_image(image: torch.Tensor, patch_size = 16):
    """
    将 (C, H, W) 形状的图像分割成 14x14=196 个 16x16 小块，并编号
    - image: (C, 224, 224) 的张量
    - patch_size: 每个小块的大小，默认 16
    """
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, "图像尺寸必须是 patch_size 的整数倍"
    
    num_patches = (H // patch_size) * (W // patch_size)

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(num_patches, C, patch_size, patch_size)
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
    H, W = patch_size * (int(np.sqrt(num_patches))), patch_size * (int(np.sqrt(num_patches)))
    patches = patches.reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)), 3, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4).reshape(3, H, W)
    return patches

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

def mask(image: torch.Tensor ,rate: float):
    """
    将图像进行随机掩码处理
    - image: (C, H, W) 的张量
    """
    indices, patches = split_image(image)
    num_patches = len(indices)
    num_mask = int(num_patches * rate)
    mask_indices = np.random.choice(indices, size=num_mask, replace=False)
    patches[mask_indices] = 0
    masked_image = merge_patches(patches, indices)
    return masked_image,mask_indices

def reorder_with_mask(image: torch.Tensor,  mask_indices: np.array):
    """
    将mask图像重新排列
    - image: (C, H, W) 的张量,其中某些patches被掩码
    - mask_indices: 被掩码的patch索引
    """
    indices, patches = split_image(image)
    keep_indices = np.setdiff1d(indices, mask_indices)
    kept_patches = patches[keep_indices]
    shuffled_order = np.random.permutation(len(kept_patches))
    reordered_kept_patches = kept_patches[shuffled_order]
    patches[keep_indices] = reordered_kept_patches
    reordered_image = merge_patches(patches, indices)
    return reordered_image



def evaluate(output: torch.Tensor, target: torch.Tensor):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)

    return loss


def test_inf(model, image: torch.Tensor, target: torch.Tensor):
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


def f(new_order: np.array, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """

    model, image, target = kwargs["model"], kwargs["image"], kwargs["target"]
    reordered_image = reorder(image, new_order)
    output = test_inf(model, reordered_image, target)
    loss = evaluate(output, target)

    var = np.arange(196)
    mismatch_num = np.sum(new_order != var)
    
    # return loss + 0.1 * mismatch_num
    return loss


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


def solve(model, image: torch.Tensor, image_path: str, target: torch.Tensor, index: int, output: torch.Tensor, log_directory: str, search_times=10):
    """
    优化问题求解
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    - index: 图像索引
    """    
    cnt = 0
    while cnt < search_times:
        cnt += 1
        algorithm_param = {'max_num_iteration': 500,\
                    'population_size':100,\
                    'mutation_probability':0.2,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.8,\
                    'parents_portion': 0.3,\
                    'crossover_type':'pmx',\
                    'mutation_type':'swap',\
                    'init_type':'random',\
                    'max_iteration_without_improv':50,\
                    'improvement_threshold': 0.01}

        m=ga(function=f,\
                dimension=196,\
                function_timeout=10000,\
                algorithm_parameters=algorithm_param,\
                log_limit=100,\
                log_directory=f'{log_directory}/{index}/{cnt}',\
                model=model,\
                image=image,\
                target=target)

        m.run()

        # 保存算法参数
        if not os.path.exists(f'{log_directory}/algorithm_parameter'):
            with open(f'{log_directory}/algorithm_parameter', 'w') as file:
                for key, value in algorithm_param.items():
                    file.write(f'{key}: {value}\n')

        # 保存每次搜索结果
        dir_path = f'{log_directory}/{index}/{cnt}'
        os.makedirs(dir_path, exist_ok=True)
        np.savetxt(f'{dir_path}/logits.npy', output.cpu().detach().numpy().T)
        var = np.arange(196)
        mismatch_num = np.sum(m.best_variable.astype(int) != var)
        with open(f'{dir_path}/label_change.txt', 'w') as file:
            file.write(f'{index}: \n\tgt_label: {target[0]} \n\tinf_label: {torch.argmax(output)} \n\tbeforeloss: {evaluate(output, target)} \n\tafter_loss: {m.best_function} \n\tmismatch_num: {mismatch_num} \n\timage_path: {image_path} \n\n')
        original_image_path = f'{dir_path}/original_image.png'
        reordered_image_path = f'{dir_path}/reordered_image.png'
        save_images(m.best_variable, image, original_image_path, reordered_image_path)


def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush()



def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")
    torch.cuda.set_device(2)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    # np.random.seed(seed)
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
        img_size=args.input_size
    )
      
    model.to(device)

    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_directory = f'/home/wangsj2/Programs/order-matters/deit/random_sample/{current_time}'
    save_directory = f'/home/wangsj2/Programs/order-matters/deit/random_mask/{current_time}'
    device_index = torch.cuda.current_device()

    device_name = torch.cuda.get_device_name(device_index)

    print(f"device {device_name} /{device_index}")
    model.eval()
    with torch.no_grad():
        cnt_acc = 0
        cnt_rec = 0
        pic_num = 50000
        for index in range(0, pic_num):
            image, target = dataset_val[index]
            image_path = dataset_val.imgs[index][0]
            image = image.to(device)
            target = torch.tensor(target)
            target = target.to(device)
            target = target.unsqueeze(0)
            num=100

            cnt=0
            out=test_inf(model, image, target)
            res=(torch.argmax(out)==target).item()
            origin_loss=evaluate(out,target)
            prob = torch.softmax(out, dim=1)[0, target.item()].item()
            max_prob = torch.max(torch.softmax(out, dim=1), dim=1)[0].item()
            
            
            
            # mask_image,mask_indices=mask(image, 0.7)
            # reordered_masked_image=reorder_with_mask(mask_image, mask_indices)
            # mask_out=test_inf(model, mask_image, target)
            # mask_res=(torch.argmax(mask_out)==target).item()
            # mask_loss=evaluate(mask_out,target)
            # reordered_masked_out=test_inf(model, reordered_masked_image, target)
            # reordered_mask_res=(torch.argmax(reordered_masked_out)==target).item()
            # reordered_mask_loss=evaluate(reordered_masked_out,target)
            # os.makedirs(f'{save_directory}/{index}', exist_ok=True)
            # vutils.save_image(image, f'{save_directory}/{index}/original_image.png')
            # vutils.save_image(mask_image, f'{save_directory}/{index}/mask_image.png')
            # vutils.save_image(reordered_masked_image, f'{save_directory}/{index}/reordered_masked_image.png')
            # with open(f'{save_directory}/{index}/random_mask.txt', 'w') as file:
            #     file.write(f'image{index} origin [{res}] loss={origin_loss}  random mask [{mask_res}] loss={mask_loss} reorder_mask[{reordered_mask_res}] loss={reordered_mask_loss}\n')

            # batch_size=128
            # for radio in[0.5,0.7,0.75]:
            #     correct_count = 0
            #     total_count = 0
            #     total_loss=0
            #     for i in range(0, num, batch_size):
            #         current_batch_size = min(batch_size, num - i)
            #         batch_images = []
            #         for _ in range(current_batch_size):
            #             mask_image,mask_indices=mask(image, radio)
            #             batch_images.append(mask_image)

            #         batch_images = torch.stack(batch_images).to(device)
            #         batch_targets = torch.full((batch_images.size(0),), target.item()).to(device)

            #         output = model(batch_images)
            #         correct_count += (torch.argmax(output, dim=1) == batch_targets).sum().item()
            #         total_count += batch_targets.size(0)
            #         total_loss+=evaluate(output,batch_targets).item()*batch_targets.size(0)

            #         progress(total_count, num, status=f'image {index} radio{radio}')
                        
                    
            #     assert total_count==num
            #     os.makedirs(save_directory, exist_ok=True)
            #     with open(f'{save_directory}/random_mask.txt', 'a') as file:
            #         file.write(f'image{index} origin [{res}] loss={origin_loss}\n \t random mask {radio} avg_loss {total_loss/total_count} correct rate {correct_count}/{total_count}  {correct_count/total_count}\n')
            
 

            
            batch_size=200
            correct_count = 0
            total_count = 0
            high_confidence_count = 0
            sampled_solutions=set()
            for i in range(0, num, batch_size):
                current_batch_size = min(batch_size, num - i)
                batch_images = []
                for _ in range(current_batch_size):
                    new_order = np.arange(196)
                    for t in range(10):
                        # 随机选择两个索引进行交换
                        idx1, idx2 = sorted(np.random.choice(196, 2, replace=False))
                        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
                    sampled_solutions.add(tuple(new_order))
                    reordered = reorder(image, new_order)
                    batch_images.append(reordered)
                    # while True:
                    #     random_order = np.random.permutation(196) 
                    #     if tuple(random_order) not in sampled_solutions:
                    #         sampled_solutions.add(tuple(random_order))
                    #         reordered = reorder(image, random_order)
                    #         batch_images.append(reordered)
                    #         break

                batch_images = torch.stack(batch_images).to(device)
                batch_targets = torch.full((batch_images.size(0),), target.item()).to(device)

                output = model(batch_images)
                correct_count += (torch.argmax(output, dim=1) == batch_targets).sum().item() #所有分类正确的解的个数
                total_count += batch_targets.size(0)
                probs = torch.softmax(output, dim=1)
                batch_prob = probs[torch.arange(probs.size(0)), batch_targets]# 取出每个样本真实类别的概率
                high_confidence_count += (batch_prob >= 0.5).sum().item() #所有置信度大于0.5的个数

                progress(total_count, num, status=f'image {index}')
                    
                
            assert total_count==num
            os.makedirs(log_directory, exist_ok=True)
            with open(f'{log_directory}/random_sample.txt', 'a') as file:
                file.write(f'{index}\t{prob:.6f}\t{max_prob-prob:.6f}\t{res}\t{correct_count/total_count:.6f}\t{high_confidence_count/total_count:.6f}\n')
            
            if res == True:
                cnt_acc += 1
            
            if correct_count / total_count >= 0.5:
                cnt_rec += 1

        with open(f'{log_directory}/random_sample.txt', 'a') as file:
            file.write(f'{cnt_acc/pic_num:.6f}\t{cnt_rec/pic_num:.6f}\n')

              
              


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)