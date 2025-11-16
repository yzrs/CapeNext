import argparse
import os
import os.path as osp
import random
import shutil
import uuid

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

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    targetCatId = 20
    dst_folder = 'tools/ap10k/{}'.format(targetCatId)
    if not osp.exists(dst_folder):
        os.makedirs(dst_folder)
    wrote_set = set()
    for ann in dataset.db:
        catId = ann['category_id']
        if catId == targetCatId:
            original_image_path = ann['image_file']
            img_name = osp.basename(original_image_path)
            if img_name in wrote_set:
                continue
            # copy image to dst folder
            dst_image_path = osp.join(dst_folder, img_name)
            shutil.copy(original_image_path, dst_image_path)
            wrote_set.add(img_name)
            print(f"Copied {img_name} to {dst_image_path}")
        else:
            continue
        
if __name__ == '__main__':
    main()
