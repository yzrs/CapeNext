import torch
from mmpose.models.backbones.hrnet import HRNet

def get_hrnet_w32():
    extra = dict(
        stage1=dict(
         num_modules=1,
         num_branches=1,
         block='BOTTLENECK',
         num_blocks=(4, ),
         num_channels=(64, )),
        stage2=dict(
         num_modules=1,
         num_branches=2,
         block='BASIC',
         num_blocks=(4, 4),
         num_channels=(32, 64)),
        stage3=dict(
         num_modules=4,
         num_branches=3,
         block='BASIC',
         num_blocks=(4, 4, 4),
         num_channels=(32, 64, 128)),
        stage4=dict(
         num_modules=3,
         num_branches=4,
         block='BASIC',
         num_blocks=(4, 4, 4, 4),
         num_channels=(32, 64, 128, 256)))
    model = HRNet(extra, in_channels=3)
    return model

# def test_hrnet():
#     model = get_hrnet_w32()
#     model.init_weights()
#     input_tensor = torch.rand(1, 3, 256, 256)
#     output = model(input_tensor)
#     print(output[0].shape)
#
# if __name__ == '__main__':
#     test_hrnet()
# init_cfg=dict(
#     type='Pretrained',
#     checkpoint='https://download.openmmlab.com/mmpose/'
#     'pretrain_models/hrnet_w32-36af842e.pth')