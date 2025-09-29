# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    '../_base_/models/upernet_r50.py',
    "../_base_/datasets/nudt_sirst.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

# 修改 norm_cfg 使用 GroupNorm 避免 BatchNorm 问题
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
deepspeed = False
deepspeed_config = 'zero_configs/adam_zero1_fp16.json'
pretrained = './pretrained/InternVL3-1B/model.safetensors'
model = dict(
    pretrained=None,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            _delete_=True,
            type='SoftIoULoss',
            smooth=1.0,
            loss_weight=1.0
        )),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg
    ),
    # decode_head=dict(
    #     _delete_=True,
    #     type='FCNHead',
    #     in_channels=1024,
    #     channels=1024,
    #     num_convs=0,
    #     dropout_ratio=0.0,
    #     concat_input=False,
    #     num_classes=2,
    #     with_norm=True,
    #     loss_decode=dict(
    #         type='SoftIoULoss',
    #         smooth=1.0,
    #         loss_weight=1.0
    #     )),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(340, 340))
)
optimizer = dict(_delete_=True, type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.0,
                 constructor='CustomLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=1.0))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1, workers_per_gpu=8, prefetch_factor=2, persistent_workers=True)

# GPU设置

# runner = dict(type='IterBasedRunner', max_iters=8000*15)

# CUDA memory management for stability
gpu_multithreading = False  # 禁用GPU多线程，避免内存竞争

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=2)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)
evaluation = dict(
    interval=800, 
    metric=['PdFa', 'ROC', 'mIoU'], 
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
)
# custom_hooks = [
#     dict(
#         type='ToBFloat16Hook',
#         priority=49),
# ]

# 解决DDP未使用参数问题
find_unused_parameters = True
