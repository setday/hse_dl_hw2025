from typing import Optional

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_iou


class APMetric(BaseMetric):
    def __init__(
            self,
            at: Optional[float] = 0.5,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.at = at

    def __call__(self, logits: Tensor, target: Tensor, **batch):
        iou = calc_iou(logits, target)
        if self.at is not None:
            iou = (iou >= self.at).float()
        return iou.mean().item()
    
class AP50Metric(APMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(at=0.5, *args, **kwargs)

class AP75Metric(APMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(at=0.75, *args, **kwargs)
