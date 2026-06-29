import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class MultiSampleDropout(nn.Module):
    """
    Multisample Dropout for improving generalization.
    Typically used at the final classification layer.
    """
    def __init__(self, classifier: nn.Module, num_samples: int = 5, p: float = 0.5):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.classifier = classifier
        
    def forward(self, x):
        # Average the logits from multiple dropout samples
        out = 0.
        for dropout in self.dropouts:
            out += self.classifier(dropout(x))
        return out / self.num_samples

def create_optimizer_and_scheduler(model, config, train_steps_per_epoch):
    """
    Creates AdamW optimizer and CosineAnnealingLR scheduler.
    """
    # Exclude weight decay for bias and LayerNorm/BatchNorm weights
    no_decay = ["bias", "LayerNorm.weight", "BatchNorm2d.weight", "BatchNorm2d.bias", "BatchNorm1d.weight", "BatchNorm1d.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
    
    # Total steps for Cosine Annealing
    total_steps = config.NUM_EPOCHS * train_steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    
    return optimizer, scheduler
