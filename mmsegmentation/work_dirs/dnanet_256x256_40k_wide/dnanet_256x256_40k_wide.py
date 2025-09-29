model = dict(
    type='SimpleEncoderDecoder',
    network=dict(
        type='DNANet',
        num_classes=1,
        input_channels=3,
        channel_size='three',
        backbone='resnet_18',
        deep_supervision=True),
    pretrained=None,
    align_corners=False,
    ignore_index=-100,
    loss_cfg=dict(
        loss_decode=dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0),
        deep_supervision=True,
        ds_losses_cfg=dict(
            aux_loss1=dict(
                in_indices=[0, 1, 2],
                losses=[dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0)
                        ]))))
train_cfg = dict()
test_cfg = dict(mode='whole')
dataset_type = 'NUDTSIRSTDataset'
data_root = '/home/intern/nips/datasets/wide_irstd'
reduce_zero_label = False
ignore_index = -100
background = (0, 0, 0)
train_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='DNANetTransforms', test_mode=False, base_size=256,
        crop_size=256),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(
                type='DNANetTransforms',
                test_mode=True,
                base_size=256,
                crop_size=256),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(
                type='DNANetTransforms',
                test_mode=True,
                base_size=256,
                crop_size=256),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    train=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='DNANetTransforms',
                test_mode=False,
                base_size=256,
                crop_size=256),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        data_root='/home/intern/nips/datasets/wide_irstd',
        img_dir='train/images',
        img_suffix='.png',
        ann_dir='train/masks',
        seg_map_suffix='.png',
        split_dir='original',
        split_suffix='.txt',
        test_mode=False,
        ignore_index=-100,
        reduce_zero_label=False,
        label_map=dict({255: 1})),
    val=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(
                        type='DNANetTransforms',
                        test_mode=True,
                        base_size=256,
                        crop_size=256),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/home/intern/nips/datasets/wide_irstd',
        img_dir='test/images',
        img_suffix='.png',
        ann_dir='test/masks',
        seg_map_suffix='.png',
        split_dir='original',
        split_suffix='.txt',
        test_mode=True,
        ignore_index=-100,
        reduce_zero_label=False,
        label_map=dict({255: 1})),
    test=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(
                        type='DNANetTransforms',
                        test_mode=True,
                        base_size=256,
                        crop_size=256),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/home/intern/nips/datasets/wide_irstd',
        img_dir='test/images',
        img_suffix='.png',
        ann_dir='test/masks',
        seg_map_suffix='.png',
        split_dir='original',
        split_suffix='.txt',
        test_mode=True,
        ignore_index=-100,
        reduce_zero_label=False,
        label_map=dict({255: 1})))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)
interval = 4000
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=2,
    create_symlink=False,
    save_optimizer=True)
evaluation = dict(
    interval=4000,
    gpu_collect=False,
    by_epoch=True,
    metric=['PdFa', 'ROC', 'mIoU'],
    ROC_thr=10,
    real_time=False,
    show=False)
find_unused_parameters = True
deepspeed = False
deepspeed_config = 'zero_configs/adam_zero1_minimal.json'
work_dir = './work_dirs/dnanet_256x256_40k_wide'
gpu_ids = range(0, 1)
auto_resume = False
