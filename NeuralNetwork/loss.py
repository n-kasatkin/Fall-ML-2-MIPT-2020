import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def linear_combination(x, y, alpha):
    return alpha * x + (1 - alpha) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class MyLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: list, targets: list):
        losses, metrics = {'total': torch.tensor(0.).to(logits[0])}, dict()
        losses['total'] = self.criterion(logits, targets.squeeze_())
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        metrics['accuracy'] = (prediction == targets).float().mean()
        return losses, metrics