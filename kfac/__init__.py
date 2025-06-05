"""Top-level module for K-FAC."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import kfac.assignment as assignment
import kfac.base_preconditioner as base_preconditioner
import kfac.distributed as distributed
import kfac.enums as enums
import kfac.layers as layers
import kfac.preconditioner as preconditioner
import kfac.scheduler as scheduler
import kfac.gpt_mega as gpt_mega

__version__ = importlib_metadata.version('kfac-pytorch')
