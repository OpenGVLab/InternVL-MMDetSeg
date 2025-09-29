import torch.nn as nn
import torch
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class SobelSoftIoULoss(nn.Module):
    """改进的SoftIoULoss，增强对小目标和边界的关注
    
    改进点：
    1. 加入Sobel边缘检测，对目标边界像素赋予更高权重
    2. 应用Log-Cosh平滑，提高梯度稳定性，特别适合小目标训练
    3. 增强对小目标的敏感性，解决其损失被背景稀释的问题
    """

    def __init__(
        self,
        smooth=1.0,
        loss_weight=1.0,
        class_weight=None,
        reduction='mean',
        loss_name='loss_improved_soft_iou',
        edge_weight=2.0,  # 边缘像素的权重放大倍数
        use_log_cosh=False,  # 是否使用Log-Cosh平滑
    ):
        super(SobelSoftIoULoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self._loss_name = loss_name
        self.edge_weight = edge_weight
        self.use_log_cosh = use_log_cosh
        
        # 初始化Sobel算子用于边缘检测
        self.edge_kernel = torch.tensor([[-1, -1, -1], 
                                         [-1, 8, -1], 
                                         [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def _get_edge_map(self, labels):
        """使用Sobel算子检测标签边缘，生成边缘映射"""
        # 确保输入是4D tensor: [B, 1, H, W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
        # conv2d 需要浮点输入
        labels = labels.float()

        # 将边缘检测核移动到与标签相同的设备
        edge_kernel = self.edge_kernel.to(labels.device)
        
        # 检测边缘
        edge_map = F.conv2d(labels, edge_kernel, padding=1)
        # 二值化边缘映射，超过阈值的视为边缘
        edge_map = (edge_map > 0.5).float()
        
        # 生成权重映射：边缘像素权重为edge_weight，非边缘为1.0
        weight_map = 1.0 + (self.edge_weight - 1.0) * edge_map
        return weight_map

    def forward(self, preds, labels, **kwargs):
        smooth = self.smooth
        if preds.ndim == 4:  # 图像输入 [B, C, H, W]
            preds = torch.sigmoid(preds)
            
            # 确保标签是二值的 (0, 1)
            if labels.max() > 1:
                labels = (labels > 0).float()
            
            # 确保标签维度匹配 [B, 1, H, W]
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
            
            # 获取边缘权重映射
            weight_map = self._get_edge_map(labels)
            
            # 计算加权的交并比
            intersection = (preds * labels * weight_map).sum(dim=(1, 2, 3))
            preds_sum = (preds * weight_map).sum(dim=(1, 2, 3))
            labels_sum = (labels * weight_map).sum(dim=(1, 2, 3))
            
            # 计算IoU
            iou = (intersection + smooth) / (preds_sum + labels_sum - intersection + smooth)
            loss = 1 - iou
            
            # 应用Log-Cosh平滑提高梯度稳定性
            if self.use_log_cosh:
                loss = torch.log(torch.cosh(loss))
            
            # 根据reduction参数处理损失
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
                
            return loss * self.loss_weight

        elif preds.ndim == 5:  # 序列输入 [B, T, C, H, W]
            loss_total = 0.0
            for i in range(preds.shape[1]):
                pred = preds[:, i, :, :, :]
                label = labels[:, i, :, :, :]
                
                pred = torch.sigmoid(pred)
                
                # 确保标签是二值的
                if label.max() > 1:
                    label = (label > 0).float()
                
                # 确保标签维度匹配
                if label.dim() == 3:
                    label = label.unsqueeze(1)
                
                # 获取边缘权重映射
                weight_map = self._get_edge_map(label)
                
                # 计算加权的交并比
                intersection = (pred * label * weight_map).sum(dim=(1, 2, 3))
                preds_sum = (pred * weight_map).sum(dim=(1, 2, 3))
                labels_sum = (label * weight_map).sum(dim=(1, 2, 3))
                
                iou = (intersection + smooth) / (preds_sum + labels_sum - intersection + smooth)
                loss = 1 - iou
                
                # 应用Log-Cosh平滑
                if self.use_log_cosh:
                    loss = torch.log(torch.cosh(loss))
                
                loss_total += loss.mean() if self.reduction == 'mean' else loss.sum()
            
            # 平均序列中的损失
            avg_loss = loss_total / preds.shape[1]
            return avg_loss * self.loss_weight
        
        else:
            raise ValueError(f"不支持的输入维度: {preds.ndim}")
        
    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name
