#!/bin/bash

# 直接定义数组
v100_split_index=(1)
titan_split_index=(2)
l40s_split_index=(3)
a100_split_index=(4)

# 公共参数
module_name=baseline
dir_name=${module_name}_setting104
work_dir_prefix=work_dirs/small/clip
config_prefix="configs/small_clip/small_clip_split"
batch_size=32
workers_per_gpu=32