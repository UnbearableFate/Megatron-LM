"""Helper wrappers for supported PyTorch modules."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 9):  # pragma: >=3.9 cover
    from typing import Literal
else:  # pragma: <3.9 cover
    from typing_extensions import Literal

import torch
import torch.distributed as dist

from kfac.layers.modules import LinearModuleHelper


class MegaLinearModuleHelper(LinearModuleHelper):
    """ModuleHelper for GPTNeoX layers."""

    def __init__(
        self,
        module: torch.nn.Module,
        model_parallel_group: dist.ProcessGroup | None,
    ):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
            model_parallel_group (ProcessGroup): model parallel distributed
                process group this rank belongs to. If None, it is assumed
                model parallelism size is 1 (i.e., there is no model
                parallelism).
            parallelism (str): "input" if the layer is split on the input or
                "output" if split on the output.
        """
        self.module = module
        self.model_parallel_group = model_parallel_group
        self.model_parallel_world_size = (
            1
            if self.model_parallel_group is None
            else dist.get_world_size(self.model_parallel_group)
        )