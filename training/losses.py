import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification to handle class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prevents nans when probability 0
        
        # alpha balancing
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Combines Focal Loss for label classification and Weighted Cross Entropy for explanation classification.
    """
    def __init__(self, num_explanations: int, alpha_focal=0.25, gamma=2.0, label_weight=1.0, exp_weight=0.5):
        super().__init__()
        self.label_loss_fn = FocalLoss(alpha=alpha_focal, gamma=gamma)
        # Using standard CrossEntropy, weights can be added dynamically if class distribution is known
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) 
        self.label_weight = label_weight
        self.exp_weight = exp_weight
        
    def forward(self, label_logits, exp_logits, label_targets, exp_targets):
        # Label loss
        loss_label = self.label_loss_fn(label_logits, label_targets)
        
        # Explanation loss (only compute for fake posts where explanations make sense)
        # Assuming label 1 is Fake. We compute explanation loss only where label_targets == 1
        fake_mask = (label_targets == 1.0)
        
        if fake_mask.sum() > 0 and exp_logits is not None:
            # Select only fake samples
            exp_logits_fake = exp_logits[fake_mask]
            exp_targets_fake = exp_targets[fake_mask]
            
            # exp_targets could be -1 if missing, CrossEntropy ignores -1
            loss_exp = self.exp_loss_fn(exp_logits_fake, exp_targets_fake)
        else:
            loss_exp = torch.tensor(0.0, device=label_logits.device)
            
        total_loss = self.label_weight * loss_label + self.exp_weight * loss_exp
        return total_loss, loss_label, loss_exp
