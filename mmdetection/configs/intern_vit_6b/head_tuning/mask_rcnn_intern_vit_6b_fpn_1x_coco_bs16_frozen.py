# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
deepspeed = False
deepspeed_config = 'zero_configs/adam_zero1_bf16.json'
pretrained = './pretrained/intern_vit_6b_224px.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternViT6B',
        pretrain_size=224,
        img_size=256,
        patch_size=16,
        embed_dim=3200,
        depth=48,
        num_heads=25,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_path_rate=0.0,
        init_values=0.1,
        with_cp=True,
        use_flash_attn=True,
        qk_normalization=True,
        layerscale_force_fp32=False,
        with_fpn=True,
        freeze_vit=True,
        out_indices=[11, 23, 35, 47],
        window_attn=[True] * 48,
        window_size=[16] * 48,
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[3200, 3200, 3200, 3200],
        out_channels=256,
        num_outs=5))
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=48, layer_decay_rate=1.0))
optimizer_config = dict(grad_clip=None)
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, interval=1, max_keep_ckpts=2)
else:
    checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=1, save_best='auto')
custom_hooks = [
    dict(
        type='ToBFloat16Hook',
        priority=49),
]
