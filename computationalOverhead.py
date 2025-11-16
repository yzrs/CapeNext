import argparse
import copy
import os
import os.path as osp
import random
import uuid
from tqdm import tqdm
from ptflops import get_model_complexity_info
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from models import *  # noqa
from models.datasets import build_dataset
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader
from mmpose.models import build_posenet
from copy import deepcopy
from thop import profile
import time

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', default=None, help='test config file path')
    parser.add_argument('checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--permute_keypoints',
        action='store_true',
        help='whether to randomly permute keypoints')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    uuid.UUID(int=0)

    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 12),
        dist=distributed,
        shuffle=False,
        drop_last=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    # dataloader_setting = {
    #     **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
    #     **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
    #     **cfg.data.get('test_dataloader', {})
    # }
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    cfg.model.additional_module_cfg = cfg.get('additional_module_cfg')
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 将model使用
    # device = torch.device('cuda:0')
    # model = MMDataParallel(model.to(device), device_ids=[0])
    device = torch.device('cuda:0')
    model = MMDataParallel(model.to(device), device_ids=[0])
    count_model_parameters(model)
    model.eval()

    # with torch.no_grad():
    #     for data in data_loader:
    #         _ = model(return_loss=False, **data)

    # --- 测量准备 ---
    total_time = 0.0
    start_event = None
    end_event = None

    # --- GPU 显存测量准备 (仅限 CUDA) ---
    initial_memory = 0
    peak_memory = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device) # 重置峰值显存统计
        initial_memory = torch.cuda.memory_allocated(device) # 获取初始显存占用
        # 使用 CUDA Events 进行更精确的 GPU 时间测量 (可选但推荐)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # --- 开始计时 (CPU 时间) ---
    start_time_cpu = time.time()

    # --- 开始计时 (GPU 时间) ---
    if start_event:
        start_event.record()

    # --- 遍历 DataLoader ---
    # 使用 torch.no_grad() 如果你只是想测量数据加载和前向传播，而不计算梯度
    # 这可以节省显存和计算时间
    with torch.no_grad():
        for data in tqdm(data_loader):
            _ = model(return_loss=False, **data)

    # --- (可选) 执行模型前向传播 ---
    # 如果你想把模型计算也包含在测量范围内，取消下面的注释
    # outputs = model(inputs)
    # ------------------------------

    # (可选) 可以在这里添加其他你想测量的操作

    # 为了确保测量的准确性，特别是在循环内部测量峰值显存时，
    # 可以在这里添加 torch.cuda.synchronize()，但这会减慢速度。
    # 通常在循环结束后测量峰值即可。

    # --- 结束计时和测量 ---
    # --- GPU 时间测量结束 ---
    if end_event:
        end_event.record()

    # --- CPU 时间测量结束 ---
    end_time_cpu = time.time()
    total_time_cpu = end_time_cpu - start_time_cpu

    # --- 同步 GPU 操作 (确保所有之前的 GPU 操作都已完成) ---
    if device.type == 'cuda':
        torch.cuda.synchronize(device) # 等待 GPU 完成
        # 获取峰值显存 (相对于 reset_peak_memory_stats 之后)
        peak_memory = torch.cuda.max_memory_allocated(device)
        # 计算 GPU 操作实际花费的时间
        if start_event and end_event:
            total_time_gpu = start_event.elapsed_time(end_event) / 1000.0 # elapsed_time 返回毫秒
        else:
            total_time_gpu = None # 无法使用 CUDA Events 测量
    else:
        total_time_gpu = None # CPU 上无 GPU 时间

    # --- 计算显存使用 (转换为 MB) ---
    initial_memory_mb = initial_memory / (1024 * 1024)
    peak_memory_mb = peak_memory / (1024 * 1024)
    memory_increase_mb = (peak_memory - initial_memory) / (1024 * 1024)

    # --- 打印结果 ---
    print("\n--- 测量结果 ---")
    print(f"遍历 DataLoader 的 CPU 总时间: {total_time_cpu:.4f} 秒")
    if total_time_gpu is not None:
        print(f"遍历 DataLoader 的 GPU 事件时间: {total_time_gpu:.4f} 秒")
    else:
        print("未使用 CUDA Events 测量 GPU 时间。")

    if device.type == 'cuda':
        print(f"初始 GPU 显存占用: {initial_memory_mb:.2f} MB")
        print(f"峰值 GPU 显存占用: {peak_memory_mb:.2f} MB")
        print(f"遍历过程中的显存增量: {memory_increase_mb:.2f} MB")
    else:
        print("在 CPU 上运行，不测量 GPU 显存。")

    print("-" * 20)


def count_model_parameters(model, print_details=True):
    """
    统计模型参数量（不依赖第三方库）

    参数:
        model: PyTorch模型实例
        print_details: 是否打印详细的模块参数信息

    返回:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = 0
    trainable_params = 0
    module_stats = {}  # 按模块统计参数

    # 遍历所有参数，按模块分组
    for name, param in model.named_parameters():
        # 解析模块名（例如"fc1.weight" -> 模块为"fc1"）
        module_name = ".".join(name.split(".")[:-1])
        # if not module_name.startswith('module.backbone'):
        #     continue
        param_count = param.numel()  # 参数数量

        # 更新总统计
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        # 更新模块统计
        if module_name not in module_stats:
            module_stats[module_name] = {"total": 0, "trainable": 0}
        module_stats[module_name]["total"] += param_count
        if param.requires_grad:
            module_stats[module_name]["trainable"] += param_count

    # 打印详细信息
    if print_details:
        print("=" * 60)
        print(f"{'模块名称':<30} | {'总参数':<15} | {'可训练参数':<15} | 占比")
        print("-" * 60)

        # 按参数量从大到小排序模块
        sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        for module, stats in sorted_modules:
            percentage = (stats["total"] / total_params) * 100 if total_params > 0 else 0
            # 格式化输出（左对齐，确保对齐美观）
            print(f"{module:<30} | {stats['total']:<15,} | {stats['trainable']:<15,} | {percentage:.2f}%")

        # 总览信息
        print("=" * 60)
        print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
        print(f"不可训练参数: {total_params - trainable_params:,} ({(total_params - trainable_params) / 1e6:.2f} M)")
        print(f"可训练比例: {trainable_params / total_params * 100:.2f}%")
        print("=" * 60)

    return total_params, trainable_params

        
if __name__ == '__main__':
    main()
