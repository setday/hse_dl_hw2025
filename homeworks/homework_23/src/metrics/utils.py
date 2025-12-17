from typing import List

from torch import Tensor
from timm.utils import accuracy


def calc_top_acc(targets: List[int] | Tensor, preds: List[int] | Tensor, top_k: int = 1) -> float:
    """
    Calculate the top-k accuracy.
    """
    acc = accuracy(preds, targets, topk=(top_k,))[0]
    if isinstance(acc, Tensor):
        return acc.item()
    return acc

def calc_iou(preds: Tensor, targets: Tensor) -> Tensor:
    """
    Calculate the Intersection over Union (IoU) metric.
    """
    preds = (preds > 0).long()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds & targets).float().sum(1)
    union = (preds | targets).float().sum(1)
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou
