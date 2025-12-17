from torch import Tensor

from src.metrics.base_metric import BaseMetric

from src.metrics.utils import calc_iou


class IOUMetric(BaseMetric):
    def __init__(
            self,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, **batch):
        return calc_iou(logits, target).mean().item()
