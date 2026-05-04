# Loss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """
    Standard CrossEntropy with Label Smoothing (Skeleton).
    Used as the primary loss for facial expression recognition.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / inputs.size(1)
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class SampleWeightedFocalContrastiveLoss(nn.Module):
    """
    Advanced Loss mentioned in the paper.
    Full implementation involving sample weight calculation and
    contrastive margin will be released after acceptance.
    """
    def __init__(self):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # 核心的对比学习和样本权重逻辑已脱敏
        return self.base_loss(logits, labels)

class SoftHGRLoss(nn.Module):
    """
    Soft-HGR Loss for multimodal/feature correlation.
    Full implementation is hidden for peer review.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # 核心特征相关性计算已脱敏
        return torch.tensor(0.0, requires_grad=True).to(args[0].device)