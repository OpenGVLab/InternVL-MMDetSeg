# --------------------------------------------------------
# VLM-Guided Infrared Small Target Detection
# For ICLR Submission
# --------------------------------------------------------

_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/nudt_sirst_512.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

# 核心创新1: 跨模态引导机制
multimodal_guidance = dict(
    enable=True,
    text_encoder=dict(
        type='VLMTextEncoder',
        model_name='internvl',
        prompt_templates=[
            "A small infrared target in thermal image",
            "Tiny hot object against cold background", 
            "Small bright spot in infrared scene"
        ]
    ),
    cross_modal_fusion=dict(
        type='CrossModalAttention',
        embed_dim=1024,
        num_heads=16,
        fusion_layers=[8, 16, 20]  # 选择性融合
    )
)

# 核心创新2: 尺度感知ViTAdapter
scale_aware_adapter = dict(
    type='ScaleAwareViTAdapter',
    scale_configs={
        'micro_targets': {
            'interaction_indexes': [[0, 5], [6, 11], [12, 17], [18, 23]],
            'cffn_ratio': 0.5,  # 更强的特征提取
            'deform_ratio': 0.5,
            'target_size_range': (1, 16)
        },
        'small_targets': {
            'interaction_indexes': [[2, 7], [8, 13], [14, 19], [20, 23]], 
            'cffn_ratio': 0.25,
            'deform_ratio': 0.25,
            'target_size_range': (16, 64)
        },
        'adaptive': True,  # 动态选择scale config
    }
)

deepspeed = True
deepspeed_config = 'zero_configs/adam_zero2_cpu_offload.json'  # 更高效的配置
pretrained = './pretrained/InternVL3-1B/model.safetensors'

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='VLMGuidedInternViTAdapter',
        pretrain_size=448,
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=True,  # 启用梯度检查点节省内存
        use_flash_attn=True,  # 加速注意力计算
        qk_normalization=True,
        layerscale_force_fp32=False,
        output_dtype="float32",
        last_feat=False,
        freeze_vit=False,
        only_feat_out=True,
        
        # 集成尺度感知适配器
        **scale_aware_adapter,
        
        # 集成跨模态引导
        multimodal_guidance=multimodal_guidance,
        
        qkv_bias=True,
        norm_type='layer_norm',
        with_simple_fpn=False,
        pretrained=pretrained,
        pretrained_type="safe"
    ),
    
    decode_head=dict(
        _delete_=True,
        type='MultiScaleVLMHead',  # 新的解码头
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        
        # 创新3: 多尺度语义融合
        semantic_fusion=dict(
            type='SemanticGuidedFusion',
            text_dim=1024,
            visual_dim=1024,
            fusion_method='cross_attention'
        ),
        
        # 创新4: 尺度自适应池化
        adaptive_pooling=dict(
            type='ScaleAdaptivePooling',
            pool_scales=(1, 2, 3, 6),
            scale_weights='learnable'  # 可学习的尺度权重
        ),
        
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        
        # 创新5: 多任务损失函数
        loss_decode=[
            dict(
                type='SoftIoULoss',
                smooth=1.0,
                loss_weight=1.0
            ),
            dict(
                type='FocalLoss',
                alpha=0.25,
                gamma=2.0,
                loss_weight=0.5
            ),
            dict(
                type='CrossModalConsistencyLoss',
                loss_weight=0.3
            )
        ]
    ),
    
    # 创新6: 深度监督 + 渐进式学习
    auxiliary_head=[
        dict(
            type='ProgressiveLearningHead',
            stage='early',
            in_channels=1024,
            in_index=0,
            channels=256,
            loss_weight=0.4
        ),
        dict(
            type='ProgressiveLearningHead', 
            stage='middle',
            in_channels=1024,
            in_index=1,
            channels=256,
            loss_weight=0.6
        )
    ],
    
    test_cfg=dict(_delete_=True, mode="whole")
)

# 优化器配置：分层学习率
optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.9,
        # VLM backbone使用更小学习率
        custom_keys={
            'backbone.vit': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.01),
            'decode_head': dict(lr_mult=1.0)
        }
    )
)

# 学习率调度
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-6,
    min_lr=1e-7,
    by_epoch=False
)

# 数据配置：增强few-shot学习能力
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        # 添加数据增强策略
        pipeline_extra=[
            dict(type='ThermalMixUp', alpha=0.2),  # 红外域特定的MixUp
            dict(type='ScaleJittering', scale_range=(0.8, 1.2)),
            dict(type='RandomCrop', crop_size=(512, 512)),
        ]
    )
)

# 评估配置
evaluation = dict(
    interval=500,
    metric=['PdFa', 'ROC', 'mIoU', 'F1-Score'],
    save_best='mIoU',
    rule='greater',
    
    # 添加few-shot evaluation
    few_shot_eval=dict(
        enable=True,
        shot_nums=[1, 3, 5, 10],
        eval_datasets=['NUDT-SIRST', 'IRSTD-1K']
    )
)

# 检查点配置
checkpoint_config = dict(
    _delete_=True,
    deepspeed=deepspeed,
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=5,
    save_optimizer=False,
    
    # 保存最佳模型用于论文实验
    save_best_checkpoints=True
)

# 日志配置：详细记录用于论文分析
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # 自定义hook记录跨模态注意力权重
        dict(type='CrossModalAnalysisHook', interval=500)
    ]
)

# ICLR实验专用配置
experiment_config = dict(
    # 可解释性分析
    interpretability=dict(
        attention_visualization=True,
        feature_visualization=True,
        failure_case_analysis=True
    ),
    
    # 效率分析
    efficiency_analysis=dict(
        measure_flops=True,
        measure_latency=True,
        memory_profiling=True
    ),
    
    # 泛化性实验
    generalization=dict(
        cross_dataset_eval=True,
        domain_adaptation=True,
        few_shot_learning=True
    )
)

# 自定义hooks用于详细分析
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        patience=10,
        metric='mIoU'
    ),
    dict(
        type='ModelComplexityHook',
        log_interval=1000
    ),
    dict(
        type='CrossModalAnalysisHook',
        analysis_interval=1000,
        save_attention_maps=True
    )
] 