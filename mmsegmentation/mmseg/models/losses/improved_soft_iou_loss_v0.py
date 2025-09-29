import torch.nn as nn
import numpy as np
import torch
from ..builder import LOSSES


@LOSSES.register_module()
class ImprovedSoftIoULossv0(nn.Module):
    """改进的SoftIoULoss，解决logits范围问题
    
    主要改进：
    1. 添加温度缩放，稳定logits范围
    2. 添加logits裁剪，防止极端值
    3. 添加监控功能，检测异常logits
    4. 支持可学习的温度参数
    
    Args:
        smooth (float): Smoothing factor. Defaults to 1.0.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float], optional): Weight of each class. Defaults to None.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_name (str): Name of the loss. Defaults to 'loss_improved_soft_iou'.
        use_temperature_scaling (bool): Whether to use temperature scaling. Defaults to True.
        initial_temperature (float): Initial temperature value. Defaults to 2.0.
        learnable_temperature (bool): Whether temperature is learnable. Defaults to True.
        logits_clip_range (tuple): Range for logits clipping. Defaults to (-10, 10).
        monitor_logits (bool): Whether to monitor logits range. Defaults to True.
    """

    def __init__(
        self,
        smooth=1.0,
        loss_weight=1.0,
        class_weight=None,
        reduction='mean',
        loss_name='loss_improved_soft_iou',
        # 温度缩放参数
        use_temperature_scaling=True,
        initial_temperature=2.0,
        learnable_temperature=True,
        # logits稳定性参数
        logits_clip_range=(-10, 10),
        monitor_logits=True,
        # 自适应归一化参数
        use_adaptive_normalization=False,
        normalization_epsilon=1e-8,
    ):
        super(ImprovedSoftIoULossv0, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self._loss_name = loss_name
        
        # 温度缩放
        self.use_temperature_scaling = use_temperature_scaling
        if use_temperature_scaling:
            if learnable_temperature:
                self.temperature = nn.Parameter(torch.tensor(initial_temperature))
            else:
                self.register_buffer('temperature', torch.tensor(initial_temperature))
        
        # Logits稳定性
        self.logits_clip_range = logits_clip_range
        self.monitor_logits = monitor_logits
        
        # 自适应归一化
        self.use_adaptive_normalization = use_adaptive_normalization
        self.normalization_epsilon = normalization_epsilon
        if use_adaptive_normalization:
            self.scale = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def _stabilize_logits(self, logits):
        """稳定化logits，包括监控、裁剪、归一化和温度缩放"""
        
        # 监控logits范围
        if self.monitor_logits:
            logits_min, logits_max = logits.min().item(), logits.max().item()
            if abs(logits_min) > 20 or abs(logits_max) > 20:
                print(f"Warning: Extreme logits detected - min: {logits_min:.2f}, max: {logits_max:.2f}")
                print(f"Temperature: {self.temperature.item():.4f}")

        # Logits裁剪（安全网）
        if self.logits_clip_range:
            logits = torch.clamp(logits, 
                               min=self.logits_clip_range[0], 
                               max=self.logits_clip_range[1])
        
        # 自适应归一化
        if self.use_adaptive_normalization:
            mean = logits.mean()
            std = logits.std()
            normalized_logits = (logits - mean) / (std + self.normalization_epsilon)
            logits = self.scale * normalized_logits + self.bias
        
        # 温度缩放
        if self.use_temperature_scaling:
            logits = logits / self.temperature
            
        return logits

    def forward(self, preds, labels, **kwargs):
        """前向传播"""
        
        # 稳定化logits
        preds = self._stabilize_logits(preds)
        
        smooth = self.smooth
        if preds.ndim == 4:  # image
            preds = torch.sigmoid(preds)

            # 确保标签是二值的 (0, 1)
            if labels.max() > 1:
                print(f"Fixed labels: max was {labels.unique()}")
                labels = (labels > 0).float()

            intersection = preds * labels.unsqueeze(1)  # B,H,W -> B,1,H,W
            
            loss = (intersection.sum() + smooth) / (
                preds.sum() + labels.sum() - intersection.sum() + smooth
            )
            loss = 1 - loss.mean()
            
            if loss < 0:
                print(f"preds: {preds.shape}")
                print(f"labels: {labels.shape}")
                print(f"intersection.sum(): {intersection.sum()}")
                print(f"preds.sum(): {preds.sum()}")
                print(f"labels.sum(): {labels.sum()}")
                print(f"loss: {loss}")
                assert 0
                
            return loss * self.loss_weight

        elif preds.ndim == 5:  # sequence
            loss_total = 0
            for i in range(preds.shape[1]):
                pred = preds[:, i, :, :, :]
                label = labels[:, i, :, :, :]
                intersection = pred * label
                loss = (intersection.sum() + smooth) / (
                    pred.sum() + label.sum() - intersection.sum() + smooth
                )
                loss = 1 - loss.mean()
                loss_total = loss_total + loss

            return (loss_total / preds.shape[1]) * self.loss_weight
        
    def get_temperature(self):
        """获取当前温度值"""
        if self.use_temperature_scaling:
            return self.temperature.item()
        return 1.0
    
    def get_logits_stats(self, logits):
        """获取logits统计信息"""
        return {
            'min': logits.min().item(),
            'max': logits.max().item(),
            'mean': logits.mean().item(),
            'std': logits.std().item(),
            'temperature': self.get_temperature()
        }
        
    @property
    def loss_name(self):
        """Loss Name.
        
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


# 为了向后兼容，也可以创建一个别名
@LOSSES.register_module()
class StableSoftIoULoss(ImprovedSoftIoULoss):
    """稳定版本的SoftIoULoss，ImprovedSoftIoULoss的别名"""
    pass
