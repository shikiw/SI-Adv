import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0., mode='targeted'):
        """Adversarial function on logits.

        Args:
            kappa (float, optional): min margin. Defaults to 0..
        """
        super(LogitsAdvLoss, self).__init__()

        self.kappa = kappa
        self.mode = mode

    def forward(self, logits, targets):
        """Adversarial loss function using logits.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        if self.mode.lower() == 'targeted':
            loss = torch.clamp(other_logits - real_logits + self.kappa, min=0.)
        else:
            loss = torch.clamp(real_logits - other_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self, mode='targeted'):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()
        self.mode = mode

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        if self.mode.lower() == 'targeted':
            loss = F.cross_entropy(logits, targets)
        else:
            loss = -F.cross_entropy(logits, targets)
        return loss
