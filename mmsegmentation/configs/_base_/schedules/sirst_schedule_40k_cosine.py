# optimizer = dict(type="AdamW", lr=0.0001, weight_decay=1e-5)
# optimizer = dict(type="Adam", lr=0.0005, weight_decay=1e-5, betas=(0.9, 0.999))
# optimizer = dict(type="Adam", lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.999))


optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,  # 最小学习率，避免完全归零
    warmup='linear',
    warmup_iters=2000,  # warmup迭代数，约占总训的2.5%
    warmup_ratio=0.1,   # warmup时学习率比例
    by_epoch=False
)# lr_config = dict(warmup='linear', warmup_iters=2000,
#                   warmup_by_epoch=False, policy='CosineAnnealing', min_lr=0, by_epoch=False)
# runtime settings


interval = 4000
# interval = 100
runner = dict(type="IterBasedRunner", max_iters=40000)
# todo 仅保留了后五个 checkpoint, 而不是最优的！
checkpoint_config = dict(
    by_epoch=True,
    interval=interval,
    max_keep_ckpts=1,
    create_symlink=False,
    save_optimizer=False,
)
#  metric = 'PdFa' -> ['Pd', 'Fa']
#  metric = 'ROC'  -> ['TPR', 'FPR', 'Recall', "Precision"]
#  metric = 'mIoU' -> ['mIoU', 'OA']
# 如果 real_time=True, 单个推理结果直接进行指标计算, 默认为 False, 所有结果推理结束后统一计算
evaluation = dict(
    interval=interval,
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
    gpu_collect=True,  # 使用GPU收集结果，避免CPU收集问题
    by_epoch=True,
    metric=["PdFa", "mIoU"],#    metric=["PdFa", "ROC", "mIoU"],
    ROC_thr=10,
    real_time=True,
    show=False,
)   # nan_to_num=0
find_unused_parameters = True