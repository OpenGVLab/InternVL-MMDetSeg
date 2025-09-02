# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    # '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/wide_irstd_eval_aligned.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

deepspeed = True  
deepspeed_config = 'zero_configs/adam_zero2_fp32_fast.json'  # 启用BF16 + ZeRO Stage 2，稳定且更快  

model = dict(
    type="SimpleEncoderDecoder",
    network=dict(
        type='DNANet',
        num_classes=1,  # 网络类别是 1，但是 label 中背景 0 目标 1
        input_channels=3,
        channel_size='three',  # optional = ['one', 'two', 'three', 'four']
        backbone='resnet_18',  # optional = ['resnet_10', 'resnet_18', 'resnet_34', 'vgg_10']
        deep_supervision=True,  # todo
    ),
    pretrained=None,  # 'pretrained/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar',  # 'pretrained/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar'
    align_corners=False,
    ignore_index=-100,  # 计算损失时忽略的值, 只要不是 0-num_classes 就行, 默认 - 100, 255 也没问题！！！
    loss_cfg=dict(
        # SoftIoULoss 用不到 ignore_index, CrossEntropyLoss 用的到
        loss_decode=dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0),
        # Whether to compute losses from multiple intermediate feature maps (deep supervision)
        deep_supervision=True,
        # if deep_supervision is True
        ds_losses_cfg=dict(
            aux_loss1=dict(
                in_indices=[0, 1, 2],
                losses=[
                    dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0),
                ],
            )
        ),
    ),
)

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=2)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(_delete_=True, mode="slide",crop_size=(448,448),stride=(224,224))
# test_cfg = dict(_delete_=True, mode="slide", crop_size=(128, 128), stride=(128, 128))
# evaluation = dict(
#     interval=100, )

# def load_param(channel_size='three', backbone='resnet_18'):

#     if channel_size == 'one':
#         nb_filter = [4, 8, 16, 32, 64]
#     elif channel_size == 'two':
#         nb_filter = [8, 16, 32, 64, 128]
#     elif channel_size == 'three':
#         nb_filter = [16, 32, 64, 128, 256]
#     elif channel_size == 'four':
#         nb_filter = [32, 64, 128, 256, 512]

#     if backbone == 'resnet_10':
#         num_blocks = [1, 1, 1, 1]
#     elif backbone == 'resnet_18':
#         num_blocks = [2, 2, 2, 2]
#     elif backbone == 'resnet_34':
#         num_blocks = [3, 4, 6, 3]
#     elif backbone == 'vgg_10':
#         num_blocks = [1, 1, 1, 1]

#     return nb_filter, num_blocks
evaluation = dict(
    # interval=10, 
    metric=["PdFa",'mIoU'], 
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
)
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.01, betas=(0.9, 0.999))
