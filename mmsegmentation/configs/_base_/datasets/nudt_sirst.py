# dataset settings
dataset_type = "NUDTSIRSTDataset"
data_root = "/home/intern/nips/datasets/NUDT-SIRST"
split_dir = "50_50"
reduce_zero_label = False
ignore_index = 0  # not use
background = (0, 0, 0)  # BGR

train_pipeline = [
    dict(type="LoadDataFromFile", flag='color'),
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

val_pipeline = [
    dict(type="LoadDataFromFile", flag='color'),
    dict(
        type="LoadAnnotations", reduce_zero_label=reduce_zero_label
    ),  # DNANetTransforms need mask!
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
    samples_per_gpu=4,  # 2GPU * 2 = 总batch_size=4，适合分布式训练
    workers_per_gpu=4,  # 16，原为0，已改为2，避免persistent_workers报错
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        img_dir="images",
        img_suffix=".png",
        ann_dir="masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir=split_dir,
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
        img_dir="images",
        img_suffix=".png",
        ann_dir="masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir=split_dir,
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
        img_dir="images",
        img_suffix=".png",
        ann_dir="masks",
        seg_map_suffix=".png",  # _label.png?
        split_dir=split_dir,
        split_suffix=".txt",
        test_mode=True,  # train or test/val
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
        label_map={255: 1},  # 将 target 的像素值 255 映射为 1
    ),
)
