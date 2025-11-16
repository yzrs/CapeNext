import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import logging


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, output_channels=768):
        super().__init__()
        # load from hugging face
        self.vit_backbone = timm.create_model(model_name, pretrained=pretrained)
        logging.info(f"Loaded model {model_name} by timm.create_model")

        # 移除 ViT 的分类头（通常是用于 ImageNet 分类）
        # 我们只需要它的特征提取能力
        self.vit_backbone.head = nn.Identity()

        self.embed_dim = self.vit_backbone.embed_dim

        # 计算补丁分辨率
        # 对于 vit_base_patch16_224，默认输入是 224x224，补丁大小是 16
        # 224 / 16 = 14，所以补丁特征图是 14x14
        self.patch_resolution = self.vit_backbone.patch_embed.img_size[0] // self.vit_backbone.patch_embed.patch_size[0]
        self.output_resolution = 224 // 16  # 确保是 16x16

        # 用于将 ViT 的嵌入维度（如 768）调整为所需的 512
        self.channel_adaptor = nn.Conv2d(self.embed_dim, output_channels, kernel_size=1)

    def forward(self, img_tensor):
        assert img_tensor.shape[2] == 224 and img_tensor.shape[3] == 224, \
            "Input image must be 224x224."

        # img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # 在 timm 的 ViT 模型中，如果 head 是 nn.Identity，它会返回展平的特征
        # 形状通常是 (B, num_patches + 1, embed_dim)
        # 其中 +1 是分类令牌 (CLS token)
        # 移除分类令牌 (CLS token)
        # 我们只关心图像补丁的特征

        features = self.vit_backbone.forward_features(img_tensor)
        patch_features = features[:, 1:]  # 形状: (B, num_patches, embed_dim)
        B = img_tensor.shape[0]
        H_out = W_out = self.output_resolution
        spatial_features = patch_features.permute(0, 2, 1).reshape(B, self.embed_dim, H_out, W_out)
        final_feature = self.channel_adaptor(spatial_features)
        return final_feature


def get_vit_backbone(pretrained=True):
    vit_extractor = ViTFeatureExtractor(model_name='vit_base_patch16_224', pretrained=pretrained, output_channels=768)
    return vit_extractor


if __name__ == '__main__':
    # --- 使用示例 ---
    # 创建一个输入张量
    img_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224

    # 实例化特征提取器
    # 'vit_base_patch16_224' 是一个常用的预训练 ViT 模型
    # 'patch16' 表示补丁大小为 16x16
    # '224' 表示其通常在 224x224 的图像上进行预训练
    vit_extractor = ViTFeatureExtractor(
        model_name='vit_base_patch16_224',
        pretrained=True,
        output_channels=768
        )
    vit_extractor.eval()  # 设置为评估模式，禁用 dropout 等

    with torch.no_grad():
        img_feature = vit_extractor(img_tensor)

    print(f"提取的 img_feature 形状: {img_feature.shape}")
    # 预期输出: 提取的 img_feature 形状: torch.Size([1, 768, 14, 14])