import argparse
import copy
import os
import os.path as osp
import random
import uuid
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
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 将model使用
    # device = torch.device('cuda:0')
    # model = MMDataParallel(model.to(device), device_ids=[0])
    device = torch.device('cuda:0')
    parallelModel = MMDataParallel(model.to(device), device_ids=[0])

    dummpy_input = {}
    with torch.no_grad():
        for data in data_loader:
            output = parallelModel(return_loss=False, **data)
            dummpy_input = {
                k: v
                for k, v in data.items()
                if k not in ['img_metas']
            }
            dummpy_input['img_metas'] = data['img_metas'].data[0]
            break

    for k,v in dummpy_input.items():
        if isinstance(v, torch.Tensor):
            dummpy_input[k] = v.to(device)
        if k in ['img_s','target_s','target_weight_s','img_q','target_q','target_weight_q']:
            ls = dummpy_input[k]
            for i in range(len(ls)):
                ls[i] = ls[i].to(device)
        if k in ['skeleton']:
            ls = dummpy_input[k]
            for i in range(len(ls)):
                innerLs = ls[i]
                for j in range(len(innerLs)):
                    innerLs[j] = innerLs[j].to(device)

    with torch.no_grad():
        test_output = model(**dummpy_input)

    # def getInput():
    #     encapulate_input = (
    #         dummpy_input['img_s'],
    #         dummpy_input['target_s'],
    #         dummpy_input['target_weight_s'],
    #         dummpy_input['img_q'],
    #         dummpy_input['target_q'],
    #         dummpy_input['target_weight_q'],
    #         dummpy_input['img_metas'],
    #         dummpy_input['skeleton']
    #     )
    #     return encapulate_input

    encapulate_input = (
        dummpy_input['img_s'],
        dummpy_input['target_s'],
        dummpy_input['target_weight_s'],
        dummpy_input['img_q'],
        dummpy_input['target_q'],
        dummpy_input['target_weight_q'],
        dummpy_input['img_metas'],
        dummpy_input['skeleton']
    )

    # 使用你的模型和准备好的样例输入元组
    macs, params = get_model_complexity_info(
        model,
        input_res=encapulate_input,
        # input_constructor=getInput,
        as_strings=True,
        print_per_layer_stat=True,  # 打印每层的统计信息
        verbose=True  # 打印详细信息
    )

    print('{:<30}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<30}  {:<8}'.format('Number of parameters:', params))

    # 通常，1 MAC (Multiply-Accumulate) 约等于 2 FLOPS (Floating Point Operations)
    # 如果想得到FLOPs，可以将MACs乘以2。
    # 例如:
    # num_macs = float(macs.split(' ')[0]) # 提取数值部分
    gflops = (num_macs * 2) / 1e9 # 转换为GFLOPS
    print('{:<30}  {:<8.2f} GFLOPS'.format('Computational complexity (FLOPS):', gflops))
    # 注意：ptflops 输出的 macs 字符串可能包含单位 (GMac, MMac)，解析时需要小心。
    # 更简单的方式是直接使用返回的数值 (如果 as_strings=False，则返回数值)

    # macs_val, params_val = get_model_complexity_info(model, example_inputs, as_strings=False)
    # gflops = (macs_val * 2) / 1e9
    # print(f"Computational complexity: {gflops:.2f} GFLOPS")



    # flops, params = profile(model, inputs=(*dummpy_input,))
    # print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    # unwrapped_model = model.module
    # flops, params = get_model_complexity_info(
    #     unwrapped_model,
    #     inputs=dummpy_input,
    #     print_per_layer_stat=False,  # Set to True if you want detailed layer-wise info
    #     ost=open('/dev/null', 'w')  # Suppress detailed layer printout if print_per_layer_stat is False
    # )

    # print(f"Model FLOPs: {flops}")
    # print(f"Model Parameters: {params}")

    print("debug")


        
if __name__ == '__main__':
    main()
