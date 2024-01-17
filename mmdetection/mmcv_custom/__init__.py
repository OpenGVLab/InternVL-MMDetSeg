# -*- coding: utf-8 -*-
import mmcv

from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor
from .layer_decay_optimizer_constructor_v2 import LayerDecayOptimizerConstructorV2
from .customized_text import CustomizedTextLoggerHook
from .checkpoint import load_checkpoint
import torch

__all__ = ['LayerDecayOptimizerConstructor',
           'LayerDecayOptimizerConstructorV2',
           'CustomizedTextLoggerHook',
           'load_checkpoint'
           ]


torch_version = float(torch.__version__[:4])
if torch_version >= 1.11:
    
    from mmcv.runner.hooks import HOOKS, Hook
    from mmcv.runner.optimizer.builder import OPTIMIZERS
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from mmdet.utils.util_distribution import ddp_factory   # noqa: F401,F403
    from mmdet.core.optimizers import Lion, Adan
    
    
    try:
        import apex
        OPTIMIZERS.register_module(apex.optimizers.FusedAdam)

        @OPTIMIZERS.register_module()
        class ZeroFusedAdamW(ZeroRedundancyOptimizer):
            def __init__(self, params, optimizer_class=apex.optimizers.FusedAdam, **kwargs):
                super().__init__(params[0]['params'],
                                 optimizer_class=optimizer_class,
                                 parameters_as_bucket_view=True,
                                 **kwargs)
                for i in range(1, len(params)):
                    self.add_param_group(params[i])
    except:
        print("please install apex for fused_adamw")
    
    
    @OPTIMIZERS.register_module()
    class ZeroAdamW(ZeroRedundancyOptimizer):
        def __init__(self, params, optimizer_class=torch.optim.AdamW, **kwargs):
            super().__init__(params[0]['params'],
                             optimizer_class=optimizer_class,
                             parameters_as_bucket_view=True,
                             **kwargs)
            for i in range(1, len(params)):
                self.add_param_group(params[i])


    @OPTIMIZERS.register_module()
    class ZeroLion(ZeroRedundancyOptimizer):
        def __init__(self, params, optimizer_class=Lion, **kwargs):
            super().__init__(params[0]['params'],
                             optimizer_class=optimizer_class,
                             parameters_as_bucket_view=True,
                             **kwargs)
            for i in range(1, len(params)):
                self.add_param_group(params[i])
    

    @OPTIMIZERS.register_module()
    class ZeroAdan(ZeroRedundancyOptimizer):
        def __init__(self, params, optimizer_class=Adan, **kwargs):
            super().__init__(params[0]['params'],
                             optimizer_class=optimizer_class,
                             parameters_as_bucket_view=True,
                             **kwargs)
            for i in range(1, len(params)):
                self.add_param_group(params[i])
                
                
    @HOOKS.register_module()
    class ZeroHook(Hook):
        def __init__(self, interval):
            self.interval = interval
            
        def after_epoch(self, runner):
            runner.optimizer.consolidate_state_dict(to=0)
        
        def after_train_iter(self, runner):
            if self.every_n_iters(runner, self.interval):
                runner.optimizer.consolidate_state_dict(to=0)
