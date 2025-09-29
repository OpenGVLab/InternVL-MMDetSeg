import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class SphericalIoULoss(nn.Module):
    """基于nGPT思想的超球面IoU Loss
    
    核心洞察：
    1. nGPT将所有向量归一化到单位超球面，获得了训练稳定性
    2. 我们将logits映射到超球面，在球面上定义IoU计算
    3. 超球面上的所有点都有相同的"能量"，避免极值问题
    
    数学原理：
    - 传统方法：logits ∈ (-∞, +∞) → sigmoid → [0,1]
    - 我们的方法：logits → normalize → S^d (单位超球面) → spherical_sigmoid → [0,1]
    
    优势：
    1. 天然避免logits极值问题
    2. 几何上更稳定的梯度流
    3. 更好的数值稳定性
    4. 保持IoU的几何直觉
    
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss. Defaults to 'loss_spherical_iou'.
        
        # 超球面参数
        sphere_dimension (int): 超球面维度. Defaults to 16.
        temperature (float): 球面sigmoid温度. Defaults to 1.0.
        
        # 投影参数
        projection_type (str): 投影方式 ['linear', 'mlp']. Defaults to 'mlp'.
        use_learnable_projection (bool): 是否使用可学习投影. Defaults to True.
        
        # IoU计算参数
        spherical_iou_type (str): 球面IoU类型 ['cosine', 'geodesic']. Defaults to 'cosine'.
        smooth (float): 平滑因子. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight=1.0,
        loss_name='loss_spherical_iou',
        
        # 超球面参数
        sphere_dimension=16,
        temperature=1.0,
        
        # 投影参数
        projection_type='mlp',
        use_learnable_projection=True,
        
        # IoU计算参数
        spherical_iou_type='cosine',
        smooth=1.0,
    ):
        super(SphericalIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
        self.sphere_dimension = sphere_dimension
        self.temperature = temperature
        self.projection_type = projection_type
        self.use_learnable_projection = use_learnable_projection
        self.spherical_iou_type = spherical_iou_type
        self.smooth = smooth
        
        # 构建投影网络
        if use_learnable_projection:
            if projection_type == 'linear':
                self.projector = nn.Linear(1, sphere_dimension)
            elif projection_type == 'mlp':
                self.projector = nn.Sequential(
                    nn.Linear(1, sphere_dimension * 2),
                    nn.ReLU(),
                    nn.Linear(sphere_dimension * 2, sphere_dimension),
                )
            else:
                raise ValueError(f"Unknown projection_type: {projection_type}")

    def project_to_sphere(self, logits):
        """将logits投影到单位超球面
        
        nGPT风格：所有向量都归一化到单位范数
        """
        B, C, H, W = logits.shape
        
        if self.use_learnable_projection:
            # 重塑为适合投影的形状
            logits_flat = logits.view(-1, 1)  # [B*C*H*W, 1]
            
            # 投影到高维空间
            projected = self.projector(logits_flat)  # [B*C*H*W, d]
            
            # L2归一化到单位超球面（nGPT的核心操作）
            sphere_points = F.normalize(projected, p=2, dim=1)
            
            # 重塑回原始形状
            sphere_points = sphere_points.view(B, C, H, W, self.sphere_dimension)
        else:
            # 简单的随机投影到球面
            # 使用logits的符号和大小信息构造球面点
            angles = torch.tanh(logits) * np.pi  # 映射到[-π, π]
            
            # 构造球面坐标（简化版）
            sphere_points = torch.zeros(B, C, H, W, self.sphere_dimension, device=logits.device)
            sphere_points[..., 0] = torch.cos(angles.squeeze(1))
            sphere_points[..., 1] = torch.sin(angles.squeeze(1))
            
            # 归一化确保在单位球面上
            sphere_points = F.normalize(sphere_points, p=2, dim=-1)
        
        return sphere_points

    def spherical_sigmoid(self, sphere_points):
        """在超球面上定义sigmoid函数
        
        将球面上的点映射到[0,1]概率
        使用球面坐标的第一个分量作为"logit"
        """
        # 提取球面坐标的第一个分量作为"activation"
        sphere_logits = sphere_points[..., 0] / self.temperature  # [B, C, H, W]
        
        # 应用sigmoid
        probs = torch.sigmoid(sphere_logits)
        
        return probs

    def cosine_iou(self, pred_sphere, target_sphere):
        """基于余弦相似度的球面IoU
        
        在超球面上，余弦相似度 = 内积（因为都是单位向量）
        """
        # 计算余弦相似度 (内积)
        cosine_sim = torch.sum(pred_sphere * target_sphere, dim=-1)  # [B, C, H, W]
        
        # 将余弦相似度转换为"重叠度"
        # cos(θ) ∈ [-1, 1] → overlap ∈ [0, 1]
        overlap = (cosine_sim + 1) / 2
        
        return overlap

    def geodesic_iou(self, pred_sphere, target_sphere):
        """基于测地线距离的球面IoU
        
        测地线距离 = arccos(内积)，更精确的球面距离
        """
        # 计算内积并限制范围避免数值问题
        inner_product = torch.sum(pred_sphere * target_sphere, dim=-1)
        inner_product = torch.clamp(inner_product, -0.999, 0.999)
        
        # 测地线距离
        geodesic_dist = torch.acos(torch.abs(inner_product))
        
        # 转换为重叠度：距离越小，重叠度越大
        max_dist = np.pi / 2  # 最大测地线距离
        overlap = 1 - geodesic_dist / max_dist
        
        return overlap

    def compute_spherical_iou(self, pred_sphere, target_sphere):
        """计算球面IoU"""
        
        if self.spherical_iou_type == 'cosine':
            intersection = self.cosine_iou(pred_sphere, target_sphere)
        elif self.spherical_iou_type == 'geodesic':
            intersection = self.geodesic_iou(pred_sphere, target_sphere)
        else:
            raise ValueError(f"Unknown spherical_iou_type: {self.spherical_iou_type}")
        
        # 计算"union"：在球面上定义为预测和目标的"覆盖范围"
        pred_magnitude = torch.norm(pred_sphere, p=2, dim=-1)  # 应该都是1
        target_magnitude = torch.norm(target_sphere, p=2, dim=-1)  # 应该都是1
        
        # 球面上的"union"定义
        union = pred_magnitude + target_magnitude - intersection
        
        # IoU计算
        iou = (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
        
        return 1 - iou  # IoU loss

    def forward(self, preds, labels, **kwargs):
        """前向传播 - 超球面IoU计算"""
        
        if preds.ndim == 4:  # image [B, C, H, W]
            # 确保标签格式正确
            if labels.max() > 1:
                labels = (labels > 0).float()
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
                
            # 1. 将预测logits投影到超球面（nGPT风格）
            pred_sphere = self.project_to_sphere(preds)  # [B, C, H, W, d]
            
            # 2. 将标签也投影到球面空间
            # 标签是{0,1}，我们需要将其映射到合理的logits再投影
            label_logits = torch.log(labels + 1e-8) - torch.log(1 - labels + 1e-8)
            target_sphere = self.project_to_sphere(label_logits)  # [B, C, H, W, d]
            
            # 3. 在超球面上计算IoU
            spherical_iou_loss = self.compute_spherical_iou(pred_sphere, target_sphere)
            
            return spherical_iou_loss * self.loss_weight
            
        else:
            raise NotImplementedError("Only 4D input (image) is supported")
    
    def get_spherical_probs(self, logits):
        """获取超球面概率（用于推理/可视化）"""
        with torch.no_grad():
            sphere_points = self.project_to_sphere(logits)
            probs = self.spherical_sigmoid(sphere_points)
            return probs
    
    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name


@LOSSES.register_module()
class nGPTStyleIoULoss(SphericalIoULoss):
    """nGPT风格的IoU Loss，专门模仿nGPT的设计理念"""
    
    def __init__(self, **kwargs):
        # nGPT风格的默认参数
        kwargs.setdefault('sphere_dimension', 32)  # 更高维度
        kwargs.setdefault('projection_type', 'mlp')  # 使用MLP投影
        kwargs.setdefault('spherical_iou_type', 'cosine')  # 余弦相似度
        kwargs.setdefault('temperature', 0.5)  # 较低温度，更sharp
        super(nGPTStyleIoULoss, self).__init__(**kwargs)


@LOSSES.register_module()
class UnitSphereIoULoss(SphericalIoULoss):
    """单位球面IoU Loss，强调几何直觉"""
    
    def __init__(self, **kwargs):
        # 几何直觉优化的参数
        kwargs.setdefault('sphere_dimension', 8)   # 适中维度，便于理解
        kwargs.setdefault('spherical_iou_type', 'geodesic')  # 真实几何距离
        kwargs.setdefault('temperature', 1.0)      # 标准温度
        super(UnitSphereIoULoss, self).__init__(**kwargs)
