# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    # '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/wide_irstd_eval_aligned.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_epoch_500.py",
]

deepspeed = True  
deepspeed_config = 'zero_configs/adam_zero2_bf16_fast.json'  # 启用BF16 + ZeRO Stage 2，稳定且更快  
# pretrained = './pretrained/InternVL2_5-1B/model.safetensors'

# pretrained = '/home/intern/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_1b_dynamic_res_2nd_finetune_full/model.safetensors'
# pretrained = '/home/intern/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/irstd/model.safetensors'
pretrained = '/home/intern/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/irstd_finetune_dataset_centroid_format_filtered/model.safetensors'
model = dict(
    pretrained=None,
    type='EncoderDecoder', 
    backbone=dict(
        type='InternViTAdapter',
        pretrain_size=448,
        img_size=448,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=True,
        use_flash_attn=True,
        qk_normalization=True,
        layerscale_force_fp32=False,
        output_dtype="bfloat16",
        last_feat=False,
        freeze_vit=True,  # 冻结backbone以加速训练
        only_feat_out=True,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.25,
        qkv_bias=True,
        norm_type='layer_norm',
        with_simple_fpn=False,
        pretrained=pretrained,pretrained_type="safe"),
    decode_head=dict(
        type='UPerHead',  # 或者 PSPHead, FPN等适合多尺度特征的head
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='SobelSoftIoULossV2',
        )),
    # Deep Supervision: 添加多个auxiliary heads

    test_cfg=dict(mode="slide", crop_size=(448, 448), stride=(224, 224))
)
# optimizer = dict(_delete_=True, type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.0,
#                  constructor='CustomLayerDecayOptimizerConstructor',
#                  paramwise_cfg=dict(num_layers=24, layer_decay_rate=1.0))
# lr_config = dict(_delete_=True, policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(samples_per_gpu=1, workers_per_gpu=8, prefetch_factor=2, persistent_workers=True)

# GPU设置

# train_cfg = dict()
# test_cfg = dict(_delete_=True, mode="whole")

# runner = dict(type='IterBasedRunner', max_iters=8000*15)

# CUDA memory management for stability
# gpu_multithreading = False  # 禁用GPU多线程，避免内存竞争

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=5000, max_keep_ckpts=1)
else:
    checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)
evaluation = dict(
    # interval=10, 
    metric=['mIoU',"PdFa"], 
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
)
# custom_hooks = [
#     dict(
#         type='ToBFloat16Hook',
#         priority=49),
# ]
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.01, betas=(0.9, 0.999))

# 添加学习率调度和warmup
# lr_config = dict(
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-6,
#     power=0.9, 
#     min_lr=1e-6, 
#     by_epoch=False
# )

# 解决DDP未使用参数问题
find_unused_parameters = False
