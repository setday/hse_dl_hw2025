from torch import Tensor
from torch.nn import BCEWithLogitsLoss


class BCELossWrapper(BCEWithLogitsLoss):
    def forward(
        self, logits: Tensor, target: Tensor, **batch
    ) -> dict[str, Tensor]:
        loss = super().forward(logits.view(-1), target.view(-1).float())

        return {"loss": loss}
