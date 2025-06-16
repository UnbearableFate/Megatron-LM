from typing import cast

import torch
from kfac.layers.modules import LinearModuleHelper
class MegaLinearModuleHelper(LinearModuleHelper):
    """ModuleHelper for torch.nn.Linear modules."""
    
    def get_grad(self) -> torch.Tensor:
        """Get formatted gradients (weight and bias) of module.

        Returns:
            gradient of shape If bias != None,
            concats bias.
        """
        g = cast(torch.Tensor, self.module.weight.main_grad)
        if self.has_bias():
            g = torch.cat(
                [g, self.module.bias.main_grad.view(-1, 1)],  # type: ignore
                1,
            )
        return g
    
    def get_bias_grad(self) -> torch.Tensor:
        """Get the gradient of the bias."""
        return cast(torch.Tensor, self.module.bias.main_grad)
    
    def get_weight_grad(self) -> torch.Tensor:
        """Get the gradient of the weight."""
        return cast(torch.Tensor, self.module.weight.main_grad)

    def set_grad(self, grad: torch.Tensor) -> None:
        """Update the gradient of the module."""
        if self.has_bias():
            weight_grad = grad[:, :-1].view(self.get_weight_grad().size())
            bias_grad = grad[:, -1:].view(self.get_bias_grad().size())
        else:
            weight_grad = grad.view(self.get_weight_grad().size())

        if self.has_bias():
            self.module.bias.main_grad = bias_grad.contiguous()
        self.module.weight.main_grad = weight_grad.contiguous()