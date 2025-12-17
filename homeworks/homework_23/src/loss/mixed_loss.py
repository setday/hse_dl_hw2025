from torch import Tensor
from torch.nn import Module


class MixedLossWrapper(Module):
    def __init__(self, losses: list[Module], weights: list[float], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.losses = losses
        self.weights = weights

    def forward(self, **batch) -> dict[str, Tensor]:
        total_loss = 0.0
        loss_dict = {}

        for loss, weight in zip(self.losses, self.weights):
            loss_output = loss(**batch)
            loss_value = loss_output["loss"]
            total_loss += weight * loss_value

            for key, value in loss_output.items():
                if key != "loss":
                    loss_dict[key] = value

        loss_dict["loss"] = total_loss
        return loss_dict
