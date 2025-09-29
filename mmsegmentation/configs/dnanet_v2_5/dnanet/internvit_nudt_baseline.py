# --------------------------------------------------------
# InternVL - Baseline Ablation Study (No Adapter, Minimal ViT)
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/nudt_sirst_512.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

# 基础配置 - 不使用任何高级特性
deepspeed = False

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        img_size=512,
        patch_size=16,
        embed_dims=768,  # 使用较小的维度以加快训练
        num_layers=12,   # 减少层数
        num_heads=12,
        mlp_ratio=4.,
        drop_path_rate=0.0,  # 不使用drop path
        out_indices=[2, 5, 8, 11],  # 输出中间层特征
        final_norm=True,
        with_cls_token=False,  # 不使用CLS token
        interpolate_mode='bicubic',
        norm_cfg=dict(type='LN', eps=1e-6),
        # 不加载预训练权重，从头训练
        init_cfg=dict(type='Normal', layer='Linear', std=0.02, bias=0.)
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=4,
        norm_cfg=dict(type='LN', eps=1e-6),
    ),
    decode_head=dict(
        _delete_=True,
        type='FCNHead',  # 使用最简单的FCN head
        in_channels=256,
        in_index=3,  # 只使用最深层特征
        channels=128,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',  # 使用标准CE loss
            use_sigmoid=True,
            loss_weight=1.0
        )
    ),
    # 不使用auxiliary head
    auxiliary_head=None,
    test_cfg=dict(_delete_=True, mode="whole")
)

# 基础优化器设置
optimizer = dict(
    type="Adam",  # 使用标准Adam
    lr=1e-3,     # 较高的学习率因为从头训练
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# 简单的学习率调度
lr_config = dict(
    policy='step',
    step=[10000, 15000],
    gamma=0.1,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    by_epoch=False
)

# 数据配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

data = dict(
    samples_per_gpu=4,  # 较大的batch size
    workers_per_gpu=2,
)

# 检查点配置
checkpoint_config = dict(
    by_epoch=False, 
    interval=2000, 
    max_keep_ckpts=2
)

# 评估配置
evaluation = dict(
    interval=1000,
    metric='mIoU',
    save_best='mIoU',
    rule='greater',
)

# 训练配置
runner = dict(type='IterBasedRunner', max_iters=20000)

# 日志配置
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ]
)

# 工作目录
work_dir = './work_dirs/internvit_nudt_baseline'

# 为消融实验添加标记
experiment_name = 'baseline_vit_no_adapter' 