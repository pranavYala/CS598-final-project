import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MaskedImputationPretrainer(nn.Module):
    def __init__(self, encoder: nn.Module, mask_prob: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.mask_prob = mask_prob
        D = encoder.output_dim
        # head for mask-flag (binary classification per feature)
        self.mask_head = nn.Linear(D, D)
        # head for value regression
        self.value_head = nn.Linear(D, D)

    def forward(self, x):
        # x: (B, T, D)
        mask = torch.rand_like(x) < self.mask_prob
        x_masked = x.clone()
        x_masked[mask] = 0
        h = self.encoder(x_masked)  # (B, D_enc)
        # expand back to time dimension
        h_exp = h.unsqueeze(1).expand(-1, x.size(1), -1)  # naive, or use per-time GRU
        mask_logits = self.mask_head(h_exp)
        values = self.value_head(h_exp)
        return mask_logits, values, mask, x

    def loss(self, preds):
        mask_logits, values, mask, x_true = preds
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask.float())
        mse = F.mse_loss(values[mask], x_true[mask])
        return bce + mse

class MultiTaskPretrainer(nn.Module):
    def __init__(self, encoder: nn.Module, task_heads: dict):
        """
        task_heads: dict of {task_name: nn.Module head}
        each head maps encoder.output_dim -> task output dims
        """
        super().__init__()
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x):
        h = self.encoder(x)  # (B, D_enc)
        outputs = {}
        for t, head in self.task_heads.items():
            outputs[t] = head(h)
        return outputs

    def loss(self, outputs, labels, exclude_task=None):
        """
        outputs: dict of task->pred
        labels: dict of task->true
        exclude_task: skip this task's loss
        """
        total = 0.0
        n = 0
        for t, pred in outputs.items():
            if t == exclude_task:
                continue
            y = labels[t]
            if y.dim()==1 or y.shape[1]==1:
                total += F.binary_cross_entropy_with_logits(pred.squeeze(), y.float())
            else:
                total += F.cross_entropy(pred, y.argmax(dim=1))
            n += 1
        return total / n

