import torch.nn as nn
import numpy as np
import torch
from ..builder import LOSSES


@LOSSES.register_module()
class ImprovedSoftIoULoss(nn.Module):
    """Improved SoftIoULoss with Focal Loss component to handle class imbalance.
    
    This loss combines Soft IoU loss with Focal loss to address the issue of
    models predicting all background due to class imbalance in datasets where
    ~20% of images have no targets.

    Args:
        smooth (float): Smoothing factor for IoU calculation. Defaults to 1.0.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float], optional): Weight of each class. Defaults to None.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_name (str): Name of the loss. Defaults to 'loss_soft_iou'.
        focal_alpha (float): Alpha parameter for focal loss. Defaults to 0.25.
        focal_gamma (float): Gamma parameter for focal loss. Defaults to 2.0.
        iou_weight (float): Weight for IoU loss component. Defaults to 0.7.
        focal_weight (float): Weight for Focal loss component. Defaults to 0.3.
        empty_image_weight (float): Per-image weight when the GT mask is empty. Defaults to 0.1.
    """

    def __init__(
        self,
        smooth=1.0,
        loss_weight=1.0,
        class_weight=None,
        reduction='mean',
        loss_name='loss_soft_iou',
        focal_alpha=0.25,
        focal_gamma=2.0,
        iou_weight=0.7,
        focal_weight=0.3,
        empty_image_weight=0.1,
    ):
        super(ImprovedSoftIoULoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self._loss_name = loss_name
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.empty_image_weight = empty_image_weight

    def focal_loss_logits(self, logits, labels):
        """Compute per-image focal loss using logits for numerical stability.

        Returns a tensor of shape [B] with mean loss per image.
        """
        labels = labels.float()

        # Ensure 4D tensors: [B, C, H, W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        # BCE with logits (no reduction)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')

        # Probabilities
        probs = torch.sigmoid(logits)

        # p_t
        p_t = probs * labels + (1 - probs) * (1 - labels)

        # alpha_t
        alpha_t = self.focal_alpha * labels + (1 - self.focal_alpha) * (1 - labels)

        # Focal factor
        focal_factor = (1 - p_t).pow(self.focal_gamma)

        loss = alpha_t * focal_factor * bce

        # Reduce to per-image mean
        reduce_dims = (1, 2, 3)
        loss_per_image = loss.mean(dim=reduce_dims)

        return loss_per_image

    def soft_iou_loss(self, preds, labels):
        """Compute per-image soft IoU loss. Returns [B]."""
        smooth = self.smooth
        
        # Ensure labels are binary (0, 1)
        labels = labels.float()
        
        # Handle dimension matching: labels [B, H, W] -> [B, 1, H, W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)  # B,H,W -> B,1,H,W
        
        # Compute intersection and union
        intersection = preds * labels  # [B, C, H, W]
        
        # Compute IoU for each sample in the batch
        reduce_dims = (1, 2, 3)  # Sum over C, H, W dimensions, keep batch dimension
        inter_sum = intersection.sum(dim=reduce_dims)  # [B]
        pred_sum = preds.sum(dim=reduce_dims)  # [B]
        label_sum = labels.sum(dim=reduce_dims)  # [B]
        
        # Compute IoU for each sample
        union_sum = pred_sum + label_sum - inter_sum
        iou = (inter_sum + smooth) / (union_sum + smooth)
        
        # Compute IoU loss
        iou_loss = 1 - iou
        return iou_loss

    def forward(self, preds, labels, **kwargs):
        if preds.ndim == 4:  # image [B, C, H, W]
            logits = preds

            # Ensure labels are binary (0, 1)
            if labels.max() > 1:
                labels = (labels > 0).float()

            # Probabilities for IoU
            probs = torch.sigmoid(logits)

            # Per-image losses
            iou_per_image = self.soft_iou_loss(probs, labels)  # [B]
            focal_per_image = self.focal_loss_logits(logits, labels)  # [B]

            # Determine which images have positive pixels
            if labels.dim() == 3:
                labels_ = labels.unsqueeze(1)
            else:
                labels_ = labels
            has_pos = (labels_.sum(dim=(1, 2, 3)) > 0).float()  # [B]

            # Per-image weights: down-weight empty images to avoid trivial background collapse
            sample_weights = torch.where(has_pos > 0, torch.ones_like(has_pos), torch.full_like(has_pos, self.empty_image_weight))

            # Normalize weights to keep average weight ~1 for stable gradients
            norm = sample_weights.mean().clamp(min=1e-6)
            sample_weights = sample_weights / norm

            total_per_image = self.iou_weight * iou_per_image + self.focal_weight * focal_per_image  # [B]
            total_loss = (total_per_image * sample_weights).mean()

            return total_loss * self.loss_weight

        elif preds.ndim == 5:  # sequence
            raise NotImplementedError("5D input not supported")
        
    @property
    def loss_name(self):
        """Loss Name.
        
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
