import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class GeometricIoULoss(nn.Module):
    """几何启发的IoU Loss - 基于流形几何的极小目标优化
    
    核心数学洞察：
    1. 传统IoU在极小目标时退化为noise-dominated metric
    2. 我们将分割问题重新表述为高维流形上的几何距离问题
    3. 使用曲率感知的度量，让小目标也能产生strong geometric signal
    
    数学原理：
    - 将pixel-wise预测视为高维概率流形上的点
    - 使用Riemannian距离替代欧几里得距离(传统IoU)
    - 引入尺度自适应的曲率项，放大小目标的几何信息
    
    关键创新：
    1. Scale-Adaptive Geometric Distance: 尺度自适应几何距离
    2. Curvature-Weighted Manifold Metric: 曲率加权流形度量  
    3. Information-Theoretic Regularization: 信息论正则化
    
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss. Defaults to 'loss_geometric_iou'.
        
        # 几何参数
        manifold_dimension (int): 流形维度. Defaults to 8.
        curvature_scale (float): 曲率缩放因子. Defaults to 2.0.
        
        # 尺度自适应参数  
        scale_sensitivity (float): 尺度敏感度. Defaults to 0.5.
        min_target_size (int): 最小目标尺寸阈值. Defaults to 100.
        
        # 信息论参数
        entropy_weight (float): 熵权重. Defaults to 0.1.
        mutual_info_weight (float): 互信息权重. Defaults to 0.2.
    """

    def __init__(
        self,
        loss_weight=1.0,
        loss_name='loss_geometric_iou',
        
        # 几何参数
        manifold_dimension=8,
        curvature_scale=2.0,
        
        # 尺度自适应参数
        scale_sensitivity=0.5,
        min_target_size=100,
        
        # 信息论参数
        entropy_weight=0.1,
        mutual_info_weight=0.2,
    ):
        super(GeometricIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
        self.manifold_dimension = manifold_dimension
        self.curvature_scale = curvature_scale
        self.scale_sensitivity = scale_sensitivity
        self.min_target_size = min_target_size
        self.entropy_weight = entropy_weight
        self.mutual_info_weight = mutual_info_weight
        
        # 学习的几何参数
        self.manifold_projector = nn.Linear(1, manifold_dimension)
        self.curvature_estimator = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def compute_scale_adaptive_weight(self, labels):
        """计算尺度自适应权重
        
        数学原理：
        对于面积为A的目标，权重 w = (A_min / A)^α
        其中α是scale_sensitivity，确保小目标获得更大权重
        """
        # 计算每个样本的目标面积
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
            
        target_areas = labels.sum(dim=(2, 3))  # [B, C]
        
        # 尺度自适应权重：小目标获得更大权重
        scale_weights = torch.pow(
            self.min_target_size / (target_areas + 1e-6), 
            self.scale_sensitivity
        )
        
        # 归一化权重，保持总体权重平衡
        scale_weights = torch.clamp(scale_weights, 0.1, 10.0)
        
        return scale_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

    def project_to_manifold(self, logits):
        """将logits投影到高维流形
        
        数学原理：
        使用非线性变换将1D logits映射到高维流形空间
        f: R → R^d, 其中d是manifold_dimension
        """
        # 重塑为适合线性层的形状
        B, C, H, W = logits.shape
        logits_flat = logits.view(B * C * H * W, 1)
        
        # 投影到高维流形
        manifold_features = self.manifold_projector(logits_flat)  # [B*C*H*W, d]
        
        # L2归一化，确保在单位超球面上
        manifold_features = F.normalize(manifold_features, p=2, dim=1)
        
        # 重塑回原始空间形状
        manifold_features = manifold_features.view(B, C, H, W, self.manifold_dimension)
        
        return manifold_features

    def compute_riemannian_distance(self, pred_manifold, target_manifold, curvature):
        """计算Riemannian流形距离
        
        数学原理：
        在curved space中，距离不是简单的欧几里得距离
        d_R(p,q) = arccos(<p,q>) * (1 + κ*curvature)
        其中κ是曲率，<p,q>是内积
        """
        # 计算流形上的内积
        inner_product = torch.sum(pred_manifold * target_manifold, dim=-1)  # [B, C, H, W]
        
        # 限制内积范围，避免数值不稳定
        inner_product = torch.clamp(inner_product, -0.999, 0.999)
        
        # 基础几何距离（球面距离）
        base_distance = torch.acos(torch.abs(inner_product))
        
        # 曲率修正项
        curvature_correction = 1 + self.curvature_scale * curvature.squeeze(1)
        
        # Riemannian距离
        riemannian_distance = base_distance * curvature_correction
        
        return riemannian_distance

    def compute_information_theoretic_terms(self, logits, labels):
        """计算信息论正则化项
        
        包括：
        1. 预测熵：鼓励confident predictions
        2. 互信息：鼓励预测与标签的相关性
        """
        probs = torch.sigmoid(logits)
        
        # 预测熵 H(P)
        pred_entropy = -(probs * torch.log(probs + 1e-8) + 
                        (1 - probs) * torch.log(1 - probs + 1e-8))
        
        # 互信息近似 I(P,Y) ≈ H(P) - H(P|Y)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        labels = labels.float()
        
        # 条件熵 H(P|Y)
        pos_entropy = -(probs * labels * torch.log(probs + 1e-8)).sum() / (labels.sum() + 1e-8)
        neg_entropy = -((1 - probs) * (1 - labels) * torch.log(1 - probs + 1e-8)).sum() / ((1 - labels).sum() + 1e-8)
        conditional_entropy = pos_entropy + neg_entropy
        
        mutual_info = pred_entropy.mean() - conditional_entropy
        
        return pred_entropy.mean(), mutual_info

    def forward(self, preds, labels, **kwargs):
        """前向传播 - 几何IoU计算"""
        
        if preds.ndim == 4:  # image [B, C, H, W]
            # 确保标签格式正确
            if labels.max() > 1:
                labels = (labels > 0).float()
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
                
            # 1. 计算尺度自适应权重
            scale_weights = self.compute_scale_adaptive_weight(labels)
            
            # 2. 估计局部曲率
            curvature = self.curvature_estimator(torch.sigmoid(preds))  # [B, 1, H, W]
            
            # 3. 投影到流形空间
            pred_manifold = self.project_to_manifold(preds)  # [B, C, H, W, d]
            
            # 创建target的流形表示（简单映射）
            target_logits = torch.log(labels + 1e-8) - torch.log(1 - labels + 1e-8)
            target_manifold = self.project_to_manifold(target_logits)
            
            # 4. 计算Riemannian距离
            riemannian_dist = self.compute_riemannian_distance(
                pred_manifold, target_manifold, curvature
            )  # [B, C, H, W]
            
            # 5. 尺度加权的几何损失
            geometric_loss = (riemannian_dist * scale_weights).mean()
            
            # 6. 信息论正则化
            pred_entropy, mutual_info = self.compute_information_theoretic_terms(preds, labels)
            
            entropy_loss = self.entropy_weight * pred_entropy
            mi_loss = -self.mutual_info_weight * mutual_info  # 最大化互信息
            
            # 总损失
            total_loss = geometric_loss + entropy_loss + mi_loss
            
            return total_loss * self.loss_weight
            
        else:
            raise NotImplementedError("Only 4D input (image) is supported")
    
    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name


@LOSSES.register_module() 
class ManifoldIoULoss(GeometricIoULoss):
    """流形IoU Loss，GeometricIoULoss的别名"""
    pass


@LOSSES.register_module()
class CurvatureAwareIoULoss(GeometricIoULoss):
    """曲率感知IoU Loss，专门针对极小目标优化的版本"""
    
    def __init__(self, **kwargs):
        # 针对极小目标的特殊参数设置
        kwargs.setdefault('scale_sensitivity', 0.8)  # 更高的尺度敏感度
        kwargs.setdefault('curvature_scale', 3.0)    # 更强的曲率效应
        kwargs.setdefault('min_target_size', 50)     # 更小的目标尺寸阈值
        super(CurvatureAwareIoULoss, self).__init__(**kwargs)
