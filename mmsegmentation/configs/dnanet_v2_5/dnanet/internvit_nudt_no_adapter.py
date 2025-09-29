# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/nudt_sirst.py",  # 使用512配置匹配InternViT
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]
deepspeed = True
deepspeed_config = 'zero_configs/adam_zero1_minimal.json'
pretrained = './pretrained/InternVL3-1B/model.safetensors'
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='InternViT',
        pretrain_size=448,
        img_size=512,  # InternViT标准输入尺寸
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=False,
        use_flash_attn=False,
        qk_normalization=True,
        layerscale_force_fp32=False,
        output_dtype="float32",
        last_feat=False,
        freeze_vit=False,  # 解冻backbone进行微调
        qkv_bias=True,
        norm_type='layer_norm',
        with_simple_fpn=False,
        pretrained=pretrained,pretrained_type="safe"),
    decode_head=dict(
        _delete_=True,
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
            type='SoftIoULoss',
            smooth=1.0,
            loss_weight=1.0
        )),
    # Deep Supervision: 添加多个auxiliary heads
    # auxiliary_head=[
    #     dict(
    #         type='FCNHead',
    #         in_channels=1024,    # ResNet stage 1 output (1/4 scale)
    #         in_index=0,
    #         channels=256,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=0.1,
    #         num_classes=1,
    #         norm_cfg=dict(type='SyncBN', requires_grad=True),
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='SoftIoULoss',
    #             smooth=1.0,
    #             loss_weight=1.0
    #         )
    #     ),
    #     dict(
    #         type='FCNHead',
    #         in_channels=1024,    # ResNet stage 2 output (1/8 scale)
    #         in_index=1,
    #         channels=256,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=0.1,
    #         num_classes=1,
    #         norm_cfg=dict(type='SyncBN', requires_grad=True),
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='SoftIoULoss',
    #             smooth=1.0,
    #             loss_weight=1.0
    #         )
    #     ),
    #     dict(
    #         type='FCNHead',
    #         in_channels=1024,   # ResNet stage 3 output (1/16 scale)
    #         in_index=2,
    #         channels=256,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=0.1,
    #         num_classes=1,
    #         norm_cfg=dict(type='SyncBN', requires_grad=True),
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='SoftIoULoss',
    #             smooth=1.0,
    #             loss_weight=1.0
    #         )
    #     )
    # ],
    test_cfg=dict(_delete_=True, mode="whole")
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

# 覆盖base配置中的checkpoint_config，添加DeepSpeed支持
checkpoint_config = dict(
    _delete_=True,  # 删除base配置中的checkpoint_config
    deepspeed=deepspeed,  # DeepSpeed checkpoint支持
    by_epoch=False, 
    interval=2000, 
    max_keep_ckpts=3,
    create_symlink=False,
    save_optimizer=False  # DeepSpeed下不保存optimizer状态
)
evaluation = dict(
    # interval=100, 
    metric=['PdFa', 'ROC', 'mIoU'], 
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
)
# custom_hooks = [
#     dict(
#         type='ToBFloat16Hook',
#         priority=49),
# ]
optimizer = dict(type="Adam", lr=0.00001, weight_decay=1e-5, betas=(0.9, 0.999))

# find_unused_parameters已在base配置中定义，无需重复
