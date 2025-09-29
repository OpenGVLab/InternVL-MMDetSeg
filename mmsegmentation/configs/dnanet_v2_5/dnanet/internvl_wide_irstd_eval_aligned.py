# --------------------------------------------------------
# InternVL Configuration - 256->512 Upsampling Strategy
# 针对256×256数据集的上采样策略，充分利用InternViT预训练权重
# --------------------------------------------------------

_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/wide_irstd_eval_aligned.py",  # 使用上采样配置
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k_cosine.py",
]

deepspeed = True  
deepspeed_config = 'zero_configs/adam_zero1_minimal.json'  # 最小化配置，让mmcv管理优化器
pretrained = './pretrained/InternVL3-1B/model.safetensors'
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='InternViTAdapter',
        pretrain_size=448,
        img_size=256,
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
        only_feat_out=True,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.25,
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
# 评测时使用小batch避免OOM，训练时保持高效率
# GPU设置

# train_cfg = dict()
# test_cfg = dict(_delete_=True, mode="whole")

# runner = dict(type='IterBasedRunner', max_iters=8000*15)

# CUDA memory management for stability
# gpu_multithreading = False  # 禁用GPU多线程，避免内存竞争

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=2)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)

# Override evaluation config from base configs to avoid duplicate key error
evaluation = dict(
#     _delete_=True,  # Delete inherited evaluation configs
    interval=10,  # Match checkpoint interval
#     metric=['mIoU'], 
    save_best='mIoU',  # 保存target类IoU最佳的模型
    rule='greater',     # 明确指定IoU.target越大越好
    by_epoch=False,
    # gpu_collect=False,
)
# custom_hooks = [
#     dict(
#         type='ToBFloat16Hook',
#         priority=49),
# ]
optimizer = dict(type="Adam", lr=0.00005, weight_decay=1e-5, betas=(0.9, 0.999))

# 解决DDP未使用参数问题
find_unused_parameters = True
