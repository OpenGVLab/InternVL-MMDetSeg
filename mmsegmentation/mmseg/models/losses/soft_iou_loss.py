import torch.nn as nn
import numpy as np
import torch
from ..builder import LOSSES


@LOSSES.register_module()
class SoftIoULoss(nn.Module):
    """SoftIoULoss.
    https://github.com/YeRen123455/Infrared-Small-labels-Detection/blob/master/model/loss.py

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        smooth=1.0,
        loss_weight=1.0,
        class_weight=None,
        reduction='mean',
        loss_name='loss_soft_iou',
    ):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, preds, labels, **kwargs):

        smooth = self.smooth
        if preds.ndim == 4:  # image
            preds = torch.sigmoid(preds)

            # 确保标签是二值的 (0, 1)
            if labels.max() > 1:
                print(f"Fixed labels: max was {labels.unique()}")
                labels = (labels > 0).float()
                
                assert 0

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
            return loss

        elif preds.ndim == 5:  # sequence
            loss_total = 0
            for i in range(preds.shape[1]):
                pred = preds[:, i, :, :, :]
                label = labels[:, i, :, :, :]
                # smooth = 1
                intersection = pred * label
                loss = (intersection.sum() + smooth) / (
                    pred.sum() + label.sum() - intersection.sum() + smooth
                )
                loss = 1 - loss.mean()
                loss_total = loss_total + loss

            return loss_total / preds.shape[1]
        
    @property
    def loss_name(self):
        """Loss Name.
        
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
