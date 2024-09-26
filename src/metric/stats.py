import torch
from torch import nn

from src.tool.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class MeanDiffRatio(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor):
        return (y_pred.mean() - y_gt.mean()) / y_gt.mean()
    

@METRIC_REGISTRY.register()
class StdDiffRatio(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor):
        return (y_pred.std() - y_gt.std()) / y_gt.std()