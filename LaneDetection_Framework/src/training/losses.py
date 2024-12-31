import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map [B, C, H, W]
            target (torch.Tensor): Ground truth segmentation map [B, H, W]
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Convert target to one-hot encoding if it's not already
        if len(target.shape) == 3:
            # Assuming target is [B, H, W] with class indices
            target = F.one_hot(target, num_classes=pred.size(1))  # [B, H, W, C]
            target = target.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Flatten predictions and target for each class
        pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
        target = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]
        
        # Calculate Dice score for each class
        intersection = (pred * target).sum(dim=2)  # [B, C]
        union = pred.sum(dim=2) + target.sum(dim=2)  # [B, C]
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Average over classes and batch
        return 1 - dice.mean()

class IoULoss(nn.Module):
    """
    IoU Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map [B, C, H, W]
            target (torch.Tensor): Ground truth segmentation map [B, H, W]
            
        Returns:
            torch.Tensor: IoU loss value
        """
        # Convert target to one-hot encoding if it's not already
        if len(target.shape) == 3:
            target = F.one_hot(target, num_classes=pred.size(1))
            target = target.permute(0, 3, 1, 2)
        
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Flatten predictions and target for each class
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        # Calculate IoU for each class
        intersection = (pred * target).sum(dim=2)
        union = (pred + target - pred * target).sum(dim=2)
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    """
    Combined Loss: Cross Entropy + Dice Loss
    """
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Combined Loss
        
        Args:
            pred (torch.Tensor): Predicted segmentation map [B, C, H, W]
            target (torch.Tensor): Ground truth segmentation map [B, H, W]
            
        Returns:
            torch.Tensor: Combined loss value
        """
        print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss