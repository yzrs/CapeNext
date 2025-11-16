# mp100_path = "../dataset/mp-100"
mp100_path = "../dataset/dense-mp-100/mp-100"

additional_module_cfg = dict()
additional_module_cfg['module_name'] = None
available_module_args = {
    "CrossAttentionLayer":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "CrossAttentionEncoder":
        {
            "model_params": {"embed_dim": 512, "num_layers": 3},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "MultiModalAttentionModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "MultiModalAttentionModuleWithoutCategoryEmbedding":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "CategoryCACAEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "ResidualCrossAttentionEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "WeightedSumCrossAttentionEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "EnhanceCategoryJointEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "MultiModalJointWithinAttentionModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "MultiModalMergeAttentionModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalMergeModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalJointModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalDoubleEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalTripleEncoder":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalWithoutCategoryTextEmbedding":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding"]
        },
    "SimpleMultiModalWithoutQueryImgEmbedding":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "MoreSimpleMultiModalModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalModuleWeightedResidual":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "WeightedResidualMultiModalWithoutImg":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMaskMultiModalModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleEnhancedMultiModalModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleAttnMultiModalModule":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalModuleConstantWeight":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalModuleWithoutSelfAttn":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleMultiModalModuleWithoutCrossAttn":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        },
    "SimpleTransformerEncoderLayer":
        {
            "model_params": {"embed_dim": 512},
            "func_args": ["image_embedding", "joint_text_embedding", "category_text_embedding"]
        }
}
additional_module_cfg['available_module_args'] = available_module_args
