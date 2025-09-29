# dataset settings for Wide IRSTD (Infrared Small Target Detection)
dataset_type = "NUDTSIRSTDataset"
data_root = "/home/intern/nips/datasets/wide_irstd"
reduce_zero_label = False
ignore_index = -100  # 改为与主配置文件一致
background = (0, 0, 0)  # BGR

# 优化的训练数据增强管道
train_pipeline = [
    dict(type="LoadDataFromFile", flag='color'),  # 使用cv2后端，更快
    dict(type="LoadAnnotations", reduce_zero_label=reduce_zero_label),
    dict(
        type="DNANetTransforms",
        test_mode=False,
        base_size=256,
        crop_size=256,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

# 优化的验证管道
val_pipeline = [
    dict(type="LoadDataFromFile", flag='color'),  # 统一使用cv2
    dict(type="LoadAnnotations", reduce_zero_label=reduce_zero_label),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(256, 256),
        # img_ratios=[0.5, 1.0, 1.5, 2.0],
        flip=False,
        transforms=[
            dict(
                type="DNANetTransforms",
                test_mode=True,
                base_size=256,
                crop_size=256,
            ),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
    # dict(
    #     type="DNANetTransforms",
    #     test_mode=True,
    #     base_size=256,
    #     crop_size=256,
    # ),
    # dict(type="ImageToTensor", keys=["img"]),
    # dict(type="CollectToList", keys=["img"]),  # 绕过 MultiScaleFlipAug 输出的 List
]

test_pipeline = val_pipeline
data = dict(
    samples_per_gpu=2,  # 减少batch size，提高稳定性
    workers_per_gpu=4,  # 增加worker数量，提升数据加载速度
    prefetch_factor=4,  # 预加载因子，减少GPU等待时间
    persistent_workers=True,  # 保持worker常驻，减少重启开销
    pin_memory=True,  # 使用页锁定内存，加速GPU传输
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        img_dir="train/images",
        img_suffix=".png",
        ann_dir="train/masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir="original",
        split_suffix=".txt",
        test_mode=False,  # train or test/val
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
        label_map={255: 1},  # 将 target 的像素值 255 映射为 1
    ),
    val=dict(
        type=dataset_type,
        pipeline=val_pipeline,
        data_root=data_root,
        img_dir="test/images",
        img_suffix=".png",
        ann_dir="test/masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir="original",
        split_suffix=".txt",
        test_mode=True,  # train or test/val
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
        label_map={255: 1},  # 将 target 的像素值 255 映射为 1
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir="test/images",
        img_suffix=".png",
        ann_dir="test/masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir="original",
        split_suffix=".txt",
        test_mode=True,  # train or test/val
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
        label_map={255: 1},  # 将 target 的像素值 255 映射为 1
    ),
)
