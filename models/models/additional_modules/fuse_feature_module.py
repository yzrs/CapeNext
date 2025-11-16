import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _get_clones(module, clone_num, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(clone_num)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(clone_num)])


class CrossAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, num_layers=3):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = _get_clones(CrossAttentionLayer(embed_dim, num_heads, dropout), num_layers)

    def forward(self, query, key, value):
        output = query
        for layer in self.layers:
            output, _ = layer(output, key, value)
        return output


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=None, dropout=0.1):
        """
        初始化 SelfAttentionLayer
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param ff_dim: Feed-Forward Network 的隐藏层维度（默认与 embed_dim 相同）
        :param dropout: Dropout 概率
        """
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.ReLU(),  # 或者 nn.ReLU()
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embedding, img_embedding):
        """
        前向传播
        :param text_embedding: 文本 embedding，形状为 [batch_size, embed_dim]
        :param img_embedding: 图像 embedding，形状为 [batch_size, embed_dim]
        :return:
            output: 自注意力输出，形状为 [batch_size, 2, embed_dim]
            attn_weights: 注意力权重，形状为 [batch_size, 2, 2]
        """
        # 将文本和图像 embedding 拼接在一起 [batch_size, 2, embed_dim]
        combined_embedding = torch.cat((text_embedding.unsqueeze(1), img_embedding.unsqueeze(1)), dim=1)
        combined_embedding = self.norm1(combined_embedding)

        # 调整维度以符合 MultiheadAttention 的输入要求 [seq_len, batch_size, embed_dim]
        combined_embedding = combined_embedding.transpose(0, 1)  # [2, batch_size, embed_dim]
        attn_output, attn_weights = self.self_attention(combined_embedding, combined_embedding, combined_embedding)

        # 调整输出维度为 [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(0, 1)  # [batch_size, 2, embed_dim]

        attn_output = combined_embedding.transpose(0, 1) + self.dropout(attn_output)
        attn_output = self.norm2(attn_output)
        ffn_output = self.ffn(attn_output)
        output = attn_output + self.dropout(ffn_output)

        return output, attn_weights


# class MultiModalAttentionModule(nn.Module):
#     # setting 16
#     def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
#         """
#         初始化 MultiModalAttentionModule
#         :param embed_dim: 输入 embedding 的维度（例如 512）
#         :param num_heads: 多头注意力的头数
#         :param dropout: Dropout 概率
#         """
#         super(MultiModalAttentionModule, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#
#         self.self_attention_layer = SelfAttentionLayer(embed_dim, num_heads, dropout=dropout)
#         self.cross_attention_layer = CrossAttentionLayer(embed_dim, num_heads, dropout=dropout)
#
#         self.non_linear = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 2),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 2, embed_dim)
#         )
#
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         # 加权融合的可学习参数
#         self.weight_category = nn.Linear(embed_dim, 1)
#         self.weight_image = nn.Linear(embed_dim, 1)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim)
#         )
#
#     def weighted_fusion(self, text_cross_attn, img_cross_attn):
#         """
#         加权融合 text_cross_attn 和 img_cross_attn
#         :param text_cross_attn: 文本交叉注意力输出，形状为 [batch_size, 100, embed_dim]
#         :param img_cross_attn: 图像交叉注意力输出，形状为 [batch_size, 100, embed_dim]
#         :return: 加权融合后的输出，形状为 [batch_size, 100, embed_dim]
#         """
#         weight_category = torch.sigmoid(self.weight_category(text_cross_attn))  # [batch_size, 100, 1]
#         weight_image = torch.sigmoid(self.weight_image(img_cross_attn))        # [batch_size, 100, 1]
#         attn_output = weight_category * text_cross_attn + weight_image * img_cross_attn  # [batch_size, 100, embed_dim]
#         return attn_output
#
#     def forward(self, category_text_embedding, query_image_embedding, joint_text_embedding):
#         """
#         前向传播
#         :param category_text_embedding: 类别文本 embedding，形状为 [batch_size, embed_dim]
#         :param query_image_embedding: 查询图像 embedding，形状为 [batch_size, embed_dim]
#         :param joint_text_embedding: 联合文本 embedding，形状为 [batch_size, 100, embed_dim]
#         :return: 输出，形状为 [batch_size, 100, embed_dim]
#         """
#         # 保存残差连接
#         residual_joint_text_embedding = joint_text_embedding  # [batch_size, 100, embed_dim]
#
#         # 自注意力层：计算 category_text_embedding 和 query_image_embedding 的自注意力
#         self_attn, _ = self.self_attention_layer(category_text_embedding, query_image_embedding)  # [batch_size, 2, embed_dim]
#         self_attn = self.non_linear(self_attn) + self_attn
#         self_attn = self.norm1(self_attn)
#         self_attn = self.dropout(self_attn)
#
#         category_text_embedding = self_attn[:, 0:1, :]  # [batch_size, 1, embed_dim]
#         query_image_embedding = self_attn[:, 1:2, :]    # [batch_size, 1, embed_dim]
#
#         # 复制拓展为 [batch_size, 100, embed_dim]
#         expanded_category_text_embedding = category_text_embedding.repeat(1, 100, 1)  # [batch_size, 100, embed_dim]
#         expanded_query_image_embedding = query_image_embedding.repeat(1, 100, 1)      # [batch_size, 100, embed_dim]
#
#         # 交叉注意力层：以 joint_text_embedding 作为 query
#         text_cross_attn = self.cross_attention_layer(query=joint_text_embedding,
#                                                      key=expanded_category_text_embedding,
#                                                      value=expanded_category_text_embedding)
#         img_cross_attn = self.cross_attention_layer(query=joint_text_embedding,
#                                                     key=expanded_query_image_embedding,
#                                                     value=expanded_query_image_embedding)
#
#         attn_output = self.weighted_fusion(text_cross_attn, img_cross_attn)  # [batch_size, 100, embed_dim]
#         attn_output = attn_output + residual_joint_text_embedding  # [batch_size, 100, embed_dim]
#         attn_output = self.norm2(attn_output)  # [batch_size, 100, embed_dim]
#         ffn_output = self.ffn(attn_output)  # [batch_size, 100, embed_dim]
#         output = ffn_output + attn_output  # [batch_size, 100, embed_dim]
#         output = self.norm3(output)  # [batch_size, 100, embed_dim]
#
#         return output


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 前馈网络，可以调整中间维度大小
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        # [seq_len, batch_size, embed_dim]
        residual_query = query
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        attn_output, attn_weights = self.attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.norm1(attn_output + residual_query.permute(1, 0, 2))
        ffn_output = self.ffn(attn_output.permute(1, 0, 2))
        ffn_output = self.dropout(ffn_output)
        output = self.norm2(ffn_output)
        return output, attn_weights

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 前馈网络，可以调整中间维度大小
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)

        attn_output, attn_weights = self.attn(q, k, v)
        q = q + self.dropout(attn_output)
        q = self.norm1(q)

        ffn_output = self.ffn(q.transpose(0, 1))
        query = query + self.dropout(ffn_output)
        query = self.norm2(query)
        return query, attn_weights

class SimpleMultiModalModule(nn.Module):
    """
    setting 29 / 30
    对原本的setting24中的模块进行了调整：
        1.删除了repeat [bz,1,embed_dim] => [bz,100,embed_dim]的步骤
        2.删除了部分冗余的残差连接设计
        3.调整了多模态embedding的加权处理
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleMultiModalModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.inter_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.text_cross_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.image_cross_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.non_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 加权融合的可学习参数
        self.weight_category = nn.Linear(embed_dim, 1)
        self.weight_image = nn.Linear(embed_dim, 1)

        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.weighted_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.img_scores = 0
        self.text_scores = 0
        self.joint_scores = 0
        self.case_num = 0

    def weighted_fusion(self, text_cross_attn, img_cross_attn):
        """
        加权融合 text_cross_attn 和 img_cross_attn
        :param text_cross_attn: 文本交叉注意力输出，形状为 [batch_size, 100, embed_dim]
        :param img_cross_attn: 图像交叉注意力输出，形状为 [batch_size, 100, embed_dim]
        :return: 加权融合后的输出，形状为 [batch_size, 100, embed_dim]
        """
        weight_category = torch.sigmoid(self.weight_category(text_cross_attn))  # [batch_size, 100, 1]
        weight_image = torch.sigmoid(self.weight_image(img_cross_attn))        # [batch_size, 100, 1]
        weighted_output = weight_category * text_cross_attn + weight_image * img_cross_attn  # [batch_size, 100, embed_dim]
        return weighted_output

    def forward(self, image_embedding, joint_text_embedding, category_text_embedding):
        """
        前向传播
        :param category_text_embedding: 类别文本 embedding，形状为 [batch_size, embed_dim]
        :param image_embedding: 查询图像 embedding，形状为 [batch_size, embed_dim]
        :param joint_text_embedding: 联合文本 embedding，形状为 [batch_size, 100, embed_dim]
        :return: 输出，形状为 [batch_size, 100, embed_dim]
        """
        # 保存残差连接
        residual_joint_text_embedding = joint_text_embedding  # [batch_size, 100, embed_dim]

        # 自注意力层：计算 category_text_embedding 和 query_image_embedding 的自注意力
        # [batch_size, 2, embed_dim]
        inter_embedding = torch.concat([category_text_embedding.unsqueeze(1), image_embedding.unsqueeze(1)],dim=1)
        inter_attn, _ = self.inter_attention_layer(query=inter_embedding,
                                                   key=inter_embedding,
                                                   value=inter_embedding)
        inter_ffn_output = self.inter_ffn(inter_attn)
        inter_attn = inter_attn + self.dropout(inter_ffn_output)
        inter_attn = self.norm1(inter_attn)

        category_text_embedding = inter_attn[:, 0:1, :]  # [batch_size, 1, embed_dim]
        query_image_embedding = inter_attn[:, 1:2, :]    # [batch_size, 1, embed_dim]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! no residual add
        text_attn, _ = self.text_cross_attention_layer(query=joint_text_embedding,
                                                       key=category_text_embedding,
                                                       value=category_text_embedding)
        img_attn, _ = self.image_cross_attention_layer(query=joint_text_embedding,
                                                       key=query_image_embedding,
                                                       value=query_image_embedding)

        # [batch_size, 100, embed_dim]
        weighted_attn = self.weighted_fusion(text_attn, img_attn)
        weighted_ffn_output = self.weighted_ffn(weighted_attn)
        weighted_attn = weighted_attn + self.dropout(weighted_ffn_output)
        weighted_attn = self.norm2(weighted_attn)

        output = residual_joint_text_embedding + self.dropout(weighted_attn)
        output = self.norm3(output)

        valid_joint_num = (~(joint_text_embedding.sum(dim=-1) == 0)).squeeze(0).sum().item()

        text_cos_score = np.mean(np.array(F.cosine_similarity(
            query_image_embedding.expand(size=[query_image_embedding.shape[0],100,query_image_embedding.shape[2]]),
            output, dim=-1)[:valid_joint_num].cpu()))

        img_cos_score = np.mean(np.array(F.cosine_similarity(
            category_text_embedding.expand(size=[category_text_embedding.shape[0],100,category_text_embedding.shape[2]]),
            output, dim=-1)[:valid_joint_num].cpu()))

        joint_cos_score = np.mean(np.array(F.cosine_similarity(
            residual_joint_text_embedding, output, dim=-1).squeeze(0)[:valid_joint_num].cpu()))

        self.text_scores = (self.text_scores * self.case_num + text_cos_score) / (self.case_num + 1)
        self.img_scores = (self.img_scores * self.case_num + img_cos_score) / (self.case_num + 1)
        self.joint_scores = (self.joint_scores * self.case_num + joint_cos_score) / (self.case_num + 1)
        self.case_num += 1
        print(" text_cos_score: {}, img_cos_score: {}, joint_cos_score: {}".format(text_cos_score, img_cos_score, joint_cos_score))
        return output

    def process(self, image_embedding, joint_text_embedding, category_text_embedding):
        return self.forward(image_embedding, joint_text_embedding, category_text_embedding)

class SimpleMultiModalDoubleEncoder(nn.Module):
    """
    setting 29 / 30
    对原本的setting24中的模块进行了调整：
        1.删除了repeat [bz,1,embed_dim] => [bz,100,embed_dim]的步骤
        2.删除了部分冗余的残差连接设计
        3.调整了多模态embedding的加权处理
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleMultiModalDoubleEncoder, self).__init__()
        self.encoder_layer1 = SimpleMultiModalModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.encoder_layer2 = SimpleMultiModalModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, image_embedding, joint_text_embedding, category_text_embedding):
        layer_embedding1 = self.encoder_layer1(image_embedding, joint_text_embedding, category_text_embedding)
        layer_embedding2 = self.encoder_layer2(image_embedding, layer_embedding1, category_text_embedding)
        return layer_embedding2

    def process(self, image_embedding, joint_text_embedding, category_text_embedding):
        return self.forward(image_embedding, joint_text_embedding, category_text_embedding)


class SimpleMultiModalTripleEncoder(nn.Module):
    """
    setting 34
    对原本的setting24中的模块进行了调整：
        1.删除了repeat [bz,1,embed_dim] => [bz,100,embed_dim]的步骤
        2.删除了部分冗余的残差连接设计
        3.调整了多模态embedding的加权处理
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleMultiModalTripleEncoder, self).__init__()
        self.encoder_layer1 = SimpleMultiModalModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.encoder_layer2 = SimpleMultiModalModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.encoder_layer3 = SimpleMultiModalModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, image_embedding, joint_text_embedding, category_text_embedding):
        layer_embedding1 = self.encoder_layer1(image_embedding, joint_text_embedding, category_text_embedding)
        layer_embedding2 = self.encoder_layer2(image_embedding, layer_embedding1, category_text_embedding)
        layer_embedding3 = self.encoder_layer2(image_embedding, layer_embedding2, category_text_embedding)
        return layer_embedding3

    def process(self, image_embedding, joint_text_embedding, category_text_embedding):
        return self.forward(image_embedding, joint_text_embedding, category_text_embedding)


class SimpleMultiModalWithoutCategoryTextEmbedding(nn.Module):
    """
    setting 35
    对原本的setting24中的模块进行了调整：
        1.删除了repeat [bz,1,embed_dim] => [bz,100,embed_dim]的步骤
        2.删除了部分冗余的残差连接设计
        3.调整了多模态embedding的加权处理
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleMultiModalWithoutCategoryTextEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.inter_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.image_cross_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.non_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 加权融合的可学习参数
        self.weight_image = nn.Linear(embed_dim, 1)

        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.weighted_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def weighted_fusion(self, img_cross_attn):
        """
        加权融合 text_cross_attn 和 img_cross_attn
        :param img_cross_attn: 图像交叉注意力输出，形状为 [batch_size, 100, embed_dim]
        :return: 加权融合后的输出，形状为 [batch_size, 100, embed_dim]
        """
        weight_image = torch.sigmoid(self.weight_image(img_cross_attn))        # [batch_size, 100, 1]
        weighted_output = weight_image * img_cross_attn  # [batch_size, 100, embed_dim]
        return weighted_output

    def forward(self, image_embedding, joint_text_embedding):
        """
        前向传播
        :param image_embedding: 查询图像 embedding，形状为 [batch_size, embed_dim]
        :param joint_text_embedding: 联合文本 embedding，形状为 [batch_size, 100, embed_dim]
        :return: 输出，形状为 [batch_size, 100, embed_dim]
        """
        # 保存残差连接
        residual_joint_text_embedding = joint_text_embedding  # [batch_size, 100, embed_dim]

        # 自注意力层：计算 category_text_embedding 和 query_image_embedding 的自注意力
        # [batch_size, 2, embed_dim]
        inter_embedding = image_embedding.unsqueeze(1)
        inter_attn, _ = self.inter_attention_layer(query=inter_embedding,
                                                   key=inter_embedding,
                                                   value=inter_embedding)
        inter_ffn_output = self.inter_ffn(inter_attn)
        inter_attn = inter_attn + self.dropout(inter_ffn_output)
        inter_attn = self.norm1(inter_attn)

        query_image_embedding = inter_attn

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! no residual add
        img_attn, _ = self.image_cross_attention_layer(query=joint_text_embedding,
                                                       key=query_image_embedding,
                                                       value=query_image_embedding)

        # [batch_size, 100, embed_dim]
        weighted_attn = self.weighted_fusion(img_attn)
        weighted_ffn_output = self.weighted_ffn(weighted_attn)
        weighted_attn = weighted_attn + self.dropout(weighted_ffn_output)
        weighted_attn = self.norm2(weighted_attn)

        output = residual_joint_text_embedding + self.dropout(weighted_attn)
        output = self.norm3(output)

        return output

    def process(self, image_embedding, joint_text_embedding):
        return self.forward(image_embedding, joint_text_embedding)


class SimpleMultiModalWithoutQueryImgEmbedding(nn.Module):
    """
    setting 35
    对原本的setting24中的模块进行了调整：
        1.删除了repeat [bz,1,embed_dim] => [bz,100,embed_dim]的步骤
        2.删除了部分冗余的残差连接设计
        3.调整了多模态embedding的加权处理
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleMultiModalWithoutQueryImgEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.inter_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.text_cross_attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.non_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 加权融合的可学习参数
        self.weight_text = nn.Linear(embed_dim, 1)

        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.weighted_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def weighted_fusion(self, text_cross_attn):
        """
        加权融合 text_cross_attn 和 img_cross_attn
        :param text_cross_attn: 图像交叉注意力输出，形状为 [batch_size, 100, embed_dim]
        :return: 加权融合后的输出，形状为 [batch_size, 100, embed_dim]
        """
        weight_text = torch.sigmoid(self.weight_text(text_cross_attn))        # [batch_size, 100, 1]
        weighted_output = weight_text * text_cross_attn  # [batch_size, 100, embed_dim]
        return weighted_output

    def forward(self, category_text_embedding, joint_text_embedding):
        """
        前向传播
        :param category_text_embedding: 查询图像 embedding，形状为 [batch_size, embed_dim]
        :param joint_text_embedding: 联合文本 embedding，形状为 [batch_size, 100, embed_dim]
        :return: 输出，形状为 [batch_size, 100, embed_dim]
        """
        # 保存残差连接
        residual_joint_text_embedding = joint_text_embedding  # [batch_size, 100, embed_dim]

        # 自注意力层：计算 category_text_embedding 和 query_image_embedding 的自注意力
        # [batch_size, 2, embed_dim]
        inter_embedding = category_text_embedding.unsqueeze(1)
        inter_attn, _ = self.inter_attention_layer(query=inter_embedding,
                                                   key=inter_embedding,
                                                   value=inter_embedding)
        inter_ffn_output = self.inter_ffn(inter_attn)
        inter_attn = inter_attn + self.dropout(inter_ffn_output)
        inter_attn = self.norm1(inter_attn)

        category_embedding = inter_attn

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! no residual add
        text_attn, _ = self.text_cross_attention_layer(query=joint_text_embedding,
                                                       key=category_embedding,
                                                       value=category_embedding)

        # [batch_size, 100, embed_dim]
        weighted_attn = self.weighted_fusion(text_attn)
        weighted_ffn_output = self.weighted_ffn(weighted_attn)
        weighted_attn = weighted_attn + self.dropout(weighted_ffn_output)
        weighted_attn = self.norm2(weighted_attn)

        output = residual_joint_text_embedding + self.dropout(weighted_attn)
        output = self.norm3(output)

        return output

    def process(self, image_embedding, joint_text_embedding, category_text_embedding):
        return self.forward(category_text_embedding, joint_text_embedding)


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        初始化 MultiModalAttentionModule
        :param embed_dim: 输入 embedding 的维度（例如 512）
        :param num_heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(SimpleTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention_layer = AttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.non_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),  # 第一层：从输入维度到隐藏层
            nn.GELU(),  # 激活函数
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(embed_dim * 4, embed_dim)  # 第二层：从隐藏层到输出维度
        )

    def forward(self, image_embedding, joint_text_embedding, category_text_embedding):
        """
        前向传播
        :param category_text_embedding: 类别文本 embedding，形状为 [batch_size, embed_dim]
        :param image_embedding: 查询图像 embedding，形状为 [batch_size, embed_dim]
        :param joint_text_embedding: 联合文本 embedding，形状为 [batch_size, 100, embed_dim]
        :return: 输出，形状为 [batch_size, 100, embed_dim]
        """
        # 保存残差连接
        residual_joint_text_embedding = joint_text_embedding  # [batch_size, 100, embed_dim]

        merged_embedding = torch.concat([image_embedding,category_text_embedding],dim=1).unsqueeze(1)
        attn_output, _ = self.attention_layer(query=merged_embedding,
                                              key=merged_embedding,
                                              value=merged_embedding)
        inter_ffn_output = self.non_linear(attn_output)
        attn_output = attn_output + self.dropout(inter_ffn_output)
        attn_output = self.norm1(attn_output)  # [bz,1,1024]
        attn_output = self.mlp_layers(attn_output)
        # [bz,1,1024] -> [bz,1,512]

        # attn_output [bz,1,512] -> [bz,joint_text_embedding.shape[1],512]
        attn_output = attn_output.expand(joint_text_embedding.shape)

        output = residual_joint_text_embedding + attn_output
        output = self.norm2(output)

        return output

    def process(self, image_embedding, joint_text_embedding, category_text_embedding):
        return self.forward(image_embedding, joint_text_embedding, category_text_embedding)



if __name__ == '__main__':
    attnLayer = CrossAttentionLayer(embed_dim=512, num_heads=8, dropout=0.1)
    support_embedding = torch.randn(8, 100, 512)
    self_attn = torch.randn(8, 2, 512)
    output, weights = attnLayer(query=support_embedding, key=self_attn, value=self_attn)
    print(output[0].shape)
