# CapeNext

This repo is the official implementation of ["CapeNext: Rethinking and Refining Dynamic Support Information for Category-Agnostic Pose Estimation"](https://arxiv.org/pdf/2511.13102) [AAAI-2026]

<a href="https://arxiv.org/pdf/2511.13102"><img src="https://img.shields.io/badge/arXiv-2511.13102-b31b1b"></a>

## Introduction

**CapeNext** is available at [arXiv](https://arxiv.org/pdf/2511.13102). It's a new framework that innovatively integrates hierarchical cross-modal interaction

with dual-stream feature refinement, enhancing the joint embedding with both class-level and instance-specific cues from

textual description and specific images. Experiments on the MP-100 dataset demonstrate that, regardless of the network

backbone, CapeNext consistently outperforms state-of-the-art CAPE methods by a large margin.

![mainFig](./assets/mainFig.png)

## Results on MP-100 dataset

| methods      | Img Backbone | CLIP Backbone | Split1    | Split2    | Split3   | Split4    | Split5    | Avg       |
| ------------ | ------------ | ------------- | --------- | --------- | -------- | --------- | --------- | --------- |
| POMNet       | ResNet-50    | -             | 84.23     | 78.25     | 78.17    | 78.68     | 79.17     | 79.70     |
| CapeFormer   | ResNet-50    | -             | 89.45     | 84.88     | 83.59    | 83.53     | 85.09     | 85.31     |
| ESCAPE       | ResNet-50    | -             | 86.89     | 82.55     | 81.25    | 81.72     | 81.32     | 82.74     |
| MetaPoint+   | ResNet-50    | -             | 90.43     | 85.59     | 84.52    | 84.34     | 85.96     | 86.17     |
| X-Pose       | ResNet-50    | ViT-Base-32   | 89.07     | 85.05     | 85.26    | 85.52     | 85.79     | 86.14     |
| SDPNet       | HRNet-32     | -             | 91.54     | 86.72     | 85.49    | 85.77     | 87.26     | 87.36     |
| GraphCape    | Swinv2-T     | -             | 91.19     | **87.81** | 85.68    | 85.87     | 85.61     | 87.23     |
| CapeX        | HRNet-w32    | ViT-Base-32   | 89.1      | 85.0      | 81.9     | 84.4      | 85.4      | 85.2      |
| CapeX        | ViT-Base-16  | ViT-Base-32   | 90.75     | 82.87     | 83.18    | 85.95     | 85.49     | 85.65     |
| CapeX        | DINOv2-ViT-S | ViT-Base-32   | 90.6      | 83.74     | 83.67    | 86.87     | 85.93     | 86.18     |
| CapeX        | Swinv2-T     | ViT-Base-32   | 91.9      | 86.97     | 84.41    | 86.13     | 88.64     | 87.61     |
| **CapeNext** | HRNet-w32    | ViT-Base-32   | 90.2      | 86.0      | 82.9     | 85.4      | 87.1      | 86.3      |
| **CapeNext** | ViT-Base-16  | ViT-Base-32   | 90.84     | 86.73     | **86.5** | 82.44     | 87.91     | 86.88     |
| **CapeNext** | DINOv2-ViT-S | ViT-Base-32   | 92.12     | 87.75     | 83.76    | **87.16** | 88.95     | 87.95     |
| **CapeNext** | Swinv2-T     | ViT-Base-32   | **92.44** | 87.31     | 85.44    | 86.47     | **90.17** | **88.37** |



## Getting Started

### Conda Environment Set-up

Please run:

```
conda env create -f capenext_env.yml
conda activate capenext
```

## MP-100 Dataset

Please follow the [official guide](https://github.com/luminxu/Pose-for-Everything/blob/main/mp100/README.md) to prepare the MP-100 dataset for training and evaluation, and organize the data structure properly.

Then, use [Pose Anything's](https://github.com/orhir/PoseAnything) updated annotation file, with all the skeleton definitions, from the following [link](https://drive.google.com/drive/folders/1uRyGB-P5Tc_6TmAZ6RnOi0SWjGq9b28T?usp=sharing).

## Training

### Backbone

Pretrained weights of Swin-Transformer-V2-Tiny are taken from this [repo](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md), in the following [link](https://drive.google.com/drive/folders/1-q4mSxlNAUwDlevc3Hm5Ij0l_2OGkrcg?usp=sharing). Pretrained weights should be placed in the `./pretrained` folder.

### Training

To train the model, run:

```
python train.py --config [path_to_config_file]  --work-dir [path_to_work_dir]
```

For example:

```shell
# capenext setting
python train.py --config configs/clip/clip_split1_config.py \
    --work-dir work_dirs/tiny/clip/capenext/split1 --cfg-options data.samples_per_gpu=32 data.workers_per_gpu=32 \
    additional_module_cfg.module_name="SimpleMultiModalModule" 
```

## Evaluation and Pretrained Models

Here we provide the evaluation results of our pretrained models on MP-100 dataset along with the config files and checkpoints:

| Setting  | Backbone    |                           split 1                            |                           split 2                            |                           split 3                            |                           split 4                            |                           split 5                            | Average |
| :------: | ----------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: |
| CapeNext | Swinv2-Tiny |                            92.44                             |                            87.31                             |                            85.44                             |                            86.47                             |                            90.17                             |  88.37  |
|          |             | [weight](https://drive.google.com/file/d/1TG_EzPqRQAo55BAox6K207Y4fO4NpmjI/view?usp=share_link) / [config](configs/clip/clip_split1_config.py) | [weight](https://drive.google.com/file/d/16iHgMRopbt6iyXYnLDPpDyvBU33kKwU3/view?usp=sharing) / [config](configs/clip/clip_split2_config.py) | [weight](https://drive.google.com/file/d/1sTfN0Q5ilTxx8-s5_udDjpzi8CxtuN5E/view?usp=share_link) / [config](configs/clip/clip_split3_config.py) | [weight](https://drive.google.com/file/d/1WEff0RwWQ36CTXad-BCpW8uawrMhkSLH/view?usp=sharing) / [config](configs/clip/clip_split4_config.py) | [weight](https://drive.google.com/file/d/1lXrDhX74OySikcA0vtvdGnMlJIv1LjRV/view?usp=sharing) / [config](configs/clip/clip_split5_config.py) |         |

### Evaluation

To evaluate the pretrained model, run:

```
python test.py [path_to_config_file] [path_to_pretrained_ckpt]
```

For example:

```shell
# capenext setting
python test.py configs/clip/clip_split1_config.py \
    work_dirs/tiny/clip/capenext/split1/split1_epoch_200.pth \
    --cfg-options additional_module_cfg.module_name="SimpleMultiModalModule" 
```

## Acknowledgement

Our code is based on code from:

 - [Pose Anything](https://github.com/orhir/PoseAnything)
 - [MMPose](https://github.com/open-mmlab/mmpose)
 - [CapeFormer](https://github.com/flyinglynx/CapeFormer)
 - [CapeX](https://github.com/matanr/capex)

