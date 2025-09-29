# --------------------------------------------------------
# ViT-Base for SIRST Detection
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/nudt_sirst.py",  # 使用512配置匹配ViT
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        img_size=512,
        patch_size=16,
        embed_dim=768,         # ViT-Base: 768维
        depth=12,          # ViT-Base: 12层
        num_heads=12,           # ViT-Base: 12个注意力头
        pretrained='./pretrained/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth',  # ViT-Base预训练权重
        mlp_ratio=4,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_last_norm=True,
        only_feat_out=True,
        ),
    
    decode_head=dict(
        _delete_=True,
        type='UPerHead',
        in_channels=[768, 768, 768, 768],   # 匹配neck输出
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,           # 适中的通道数
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='SoftIoULoss',
            smooth=1.0,
            loss_weight=1.0
        )),
    
    test_cfg=dict(_delete_=True, mode="whole")
)

# 覆盖base配置中的checkpoint_config
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False, 
    interval=2000, 
    max_keep_ckpts=3,
    create_symlink=False,
    save_optimizer=False
)

evaluation = dict(
    metric=['PdFa', 'ROC', 'mIoU'], 
    save_best='mIoU',
    rule='greater',
    # interval=100
)

# 标准ViT-Base优化器设置
optimizer = dict(type="AdamW", lr=0.00005, weight_decay=0.05, betas=(0.9, 0.999))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

# ViT-Base数据配置
data = dict(samples_per_gpu=1, workers_per_gpu=8, prefetch_factor=2, persistent_workers=True) 