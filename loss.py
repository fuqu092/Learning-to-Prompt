import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, constant):
        super(CustomLoss, self).__init__()
        self.constant = constant
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, value):
        value = value.to(outputs.device)

        ce_loss = self.cross_entropy(outputs, targets)
        cu_term = self.constant * value

        total_loss = ce_loss - cu_term

        return total_loss