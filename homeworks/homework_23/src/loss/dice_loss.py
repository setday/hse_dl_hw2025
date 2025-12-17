from torch import Tensor
from torch.nn import Module


class DiceLossWrapper(Module):
    def forward(
        self, logits: Tensor, target: Tensor, **batch
    ) -> dict[str, Tensor]:
        logits = (logits > 0).long()
        target = (target > 0).long()

        logits = logits.view(logits.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (logits & target).float().sum(1)
        union = logits.sum(1) + target.sum(1)
        loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        return {"loss": loss.mean()}
