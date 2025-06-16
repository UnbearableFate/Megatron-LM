"""Eigen decomposition preconditioning implementation."""

from __future__ import annotations

import time
from typing import Callable
from typing import cast

import torch
import torch.distributed as dist

from kfac.distributed import Future
from kfac.distributed import FutureType
from kfac.distributed import get_rank
from kfac.distributed import TorchDistributedCommunicator
from kfac.enums import AllreduceMethod
from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import ModuleHelper

class KFACEigenLayer(KFACBaseLayer):
    """KFAC layer that preconditions gradients with eigen decomposition."""

    def __init__(
        self,
        module: ModuleHelper,
        *,
        tdc: TorchDistributedCommunicator,
        allreduce_method: AllreduceMethod = AllreduceMethod.ALLREDUCE,
        factor_dtype: torch.dtype | None = None,
        grad_scaler: (
            torch.cuda.amp.GradScaler | Callable[[], float] | None
        ) = None,
        inv_dtype: torch.dtype = torch.float32,
        symmetry_aware: bool = False,
        prediv_eigenvalues: bool = False,
    ) -> None:
        """Init KFACEigenLayer.

        Args:
            module (ModuleHelper): module helper that exposes interfaces for
                getting the factors and gradients of a PyTorch module.
            tdc (TorchDistributedCommunicator): communicator object. Typically
                the communicator object should be shared by all KFACBaseLayers.
            allreduce_method (AllreduceMethod): allreduce method (default:
                AllreduceMethod.ALLREDUCE).
            factor_dtype (torch.dtype): data format to store factors in. If
                None, factors are stored in the format used in training
                (default: None).
            grad_scaler (optional): optional GradScaler or callable that
                returns the scale factor used in AMP training (default: None).
            inv_dtype (torch.dtype): data format to store inverses in.
                Inverses (or eigen decompositions) may be unstable in half-
                precision (default: torch.float32).
            symmetry_aware (bool): use symmetry aware communication method.
                This is typically more helpful when the factors are very
                large (default: False).
            prediv_eigenvalues (bool): precompute the outerproduct of eigen
                values on the worker that eigen decomposes G. This reduces
                the cost of the preconditioning stage but uses more memory
                (default: False).
        """
        super().__init__(
            module=module,
            tdc=tdc,
            allreduce_method=allreduce_method,
            factor_dtype=factor_dtype,
            grad_scaler=grad_scaler,
            inv_dtype=inv_dtype,
            symmetry_aware=symmetry_aware,
        )
        self.prediv_eigenvalues = prediv_eigenvalues

        # Eigen state variables
        # Eigenvectors of self.a_factor
        self._qa: torch.Tensor | FutureType | None = None
        # Eigenvectors of self.g_factor
        self._qg: torch.Tensor | FutureType | None = None
        # Eigenvalues of self.a_factor
        self._da: torch.Tensor | FutureType | None = None
        # Eigenvalues of self.g_factor
        self._dg: torch.Tensor | FutureType | None = None
        # Outer product + damping of eigenvalues
        # Only used if self.prediv_eigenvalues
        self._dgda: torch.Tensor | FutureType | None = None

    @property
    def qa(self) -> torch.Tensor | None:
        """Get eigen vectors of A."""
        if isinstance(self._qa, Future):
            self._qa = cast(torch.Tensor, self._qa.wait())
        return self._qa

    @qa.setter
    def qa(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen vectors of A."""
        self._qa = value

    @property
    def qg(self) -> torch.Tensor | None:
        """Get eigen vectors of G."""
        if isinstance(self._qg, Future):
            self._qg = cast(torch.Tensor, self._qg.wait())
        return self._qg

    @qg.setter
    def qg(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen vectors of G."""
        self._qg = value

    @property
    def da(self) -> torch.Tensor | None:
        """Get eigen values of A."""
        if isinstance(self._da, Future):
            self._da = cast(torch.Tensor, self._da.wait())
        return self._da

    @da.setter
    def da(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen values of A."""
        self._da = value

    @property
    def dg(self) -> torch.Tensor | None:
        """Get eigen values of G."""
        if isinstance(self._dg, Future):
            self._dg = cast(torch.Tensor, self._dg.wait())
        return self._dg

    @dg.setter
    def dg(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen values of G."""
        self._dg = value

    @property
    def dgda(self) -> torch.Tensor | None:
        """Get precomputed eigen values for preconditioning."""
        if isinstance(self._dgda, Future):
            self._dgda = cast(torch.Tensor, self._dgda.wait())
        return self._dgda

    @dgda.setter
    def dgda(self, value: torch.Tensor | FutureType | None) -> None:
        """Set precomputed eigen values for preconditioning."""
        self._dgda = value

    def get_factor(self, factor_name: str) -> torch.Tensor:
        """Get factor by name.

        Args:
            factor_name (str): name of factor to get.

        Returns:
            torch.Tensor: factor tensor.
        """
        if factor_name == 'A':
            return self.a_factor
        elif factor_name == 'G':
            return self.g_factor
        elif factor_name == 'qa':
            return self.qa
        elif factor_name == 'qg':
            return self.qg
        elif factor_name == 'da':
            return self.da
        elif factor_name == 'dg':
            return self.dg
        elif factor_name == 'dgda':
            return self.dgda
        else:
            raise ValueError(f'Unknown factor name: {factor_name}')

    def set_factor(self, factor_name: str, value: torch.Tensor) -> None:
        """Set factor by name.

        Args:
            factor_name (str): name of factor to set.
            value (torch.Tensor): value to set factor to.
        """
        if factor_name == 'A':
            self.a_factor = value
        elif factor_name == 'G':
            self.g_factor = value
        elif factor_name == 'qa':
            self.qa = value
        elif factor_name == 'qg':
            self.qg = value
        elif factor_name == 'da':
            self.da = value
        elif factor_name == 'dg':
            self.dg = value
        elif factor_name == 'dgda':
            self.dgda = value
        else:
            raise ValueError(f'Unknown factor name: {factor_name}')

    def memory_usage(self) -> dict[str, int]:
        """Get memory usage for all variables in the layer."""
        sizes = super().memory_usage()
        a_size = (
            self.qa.nelement() * self.qa.element_size()
            if self.qa is not None
            else 0
        )
        a_size += (
            self.da.nelement() * self.da.element_size()
            if self.da is not None
            else 0
        )
        g_size = (
            self.qg.nelement() * self.qg.element_size()
            if self.qg is not None
            else 0
        )
        g_size += (
            self.dg.nelement() * self.dg.element_size()
            if self.dg is not None
            else 0
        )
        g_size += (
            self.dgda.nelement() * self.dgda.element_size()
            if self.dgda is not None
            else 0
        )
        sizes['a_inverses'] = a_size
        sizes['g_inverses'] = g_size
        return sizes

    def broadcast_a_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initiate A inv broadcast and store future to result.

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.

        Args:
            src (int): src rank that computed A inverse.
            group (ProcessGroup): process group to which src should broadcast
                A inv. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        ### RPC communication
        """
        if rpc_dist.global_communicator is not None and self.name not in rpc_dist.global_communicator.current_inverse_computation_layers:
            return
        if rpc_dist.global_communicator is not None :
            rpc_dist.global_communicator.send_kfac_eigen_tensor(layer_name=self.name, q=self.qa, d=self.da, dd=self.dgda, factor_type='A')
            return
        """

        if self.qa is None or (
            not self.prediv_eigenvalues and self.da is None
        ):
            if get_rank() == src:
                raise RuntimeError(
                    f'Attempt to broadcast A inv from src={src} but this rank '
                    'has not computed A inv yet.',
                )
            assert isinstance(self.a_factor, torch.Tensor)
            self.qa = torch.empty(
                self.a_factor.shape,
                device=self.a_factor.device,
                dtype=self.inv_dtype,
            )
            self.da = torch.empty(
                self.a_factor.shape[0],
                device=self.a_factor.device,
                dtype=self.inv_dtype,
            )
        self.qa = self.tdc.broadcast(  # type: ignore
            self.qa,
            src=src,
            group=group,
        )
        if not self.prediv_eigenvalues:
            assert self.da is not None
            self.da = self.tdc.broadcast(  # type: ignore
                self.da,
                src=src,
                group=group,
            )

    def broadcast_g_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initiate G inv broadcast and store future to result.

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.

        Args:
            src (int): src rank that computed G inverse.
            group (ProcessGroup): process group to which src should broadcast
                G inv. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        ### RPC communication
        """
        if rpc_dist.global_communicator is not None and rpc_dist.global_communicator.skip_inverse_computation_flag:
            return
        if rpc_dist.global_communicator is not None :
            rpc_dist.global_communicator.send_kfac_eigen_tensor(layer_name=self.name, q=self.qg, d=self.dg, dd=self.dgda,
                                                                factor_type='G')
            return
        """

        if (
            self.qg is None
            or (not self.prediv_eigenvalues and self.dg is None)
            or (self.prediv_eigenvalues and self.dgda is None)
        ):
            if get_rank() == src:
                raise RuntimeError(
                    f'Attempt to broadcast G inv from src={src} but this rank '
                    'has not computed G inv yet.',
                )
            assert isinstance(self.g_factor, torch.Tensor)
            self.qg = torch.empty(
                self.g_factor.shape,
                device=self.g_factor.device,
                dtype=self.inv_dtype,
            )
            if not self.prediv_eigenvalues:
                self.dg = torch.empty(
                    self.g_factor.shape[0],
                    device=self.g_factor.device,
                    dtype=self.inv_dtype,
                )
            else:
                assert isinstance(self.a_factor, torch.Tensor)
                self.dgda = torch.empty(
                    (self.g_factor.shape[0], self.a_factor.shape[0]),
                    device=self.g_factor.device,
                    dtype=self.inv_dtype,
                )

        self.qg = self.tdc.broadcast(  # type: ignore
            self.qg,
            src=src,
            group=group,
        )
        if not self.prediv_eigenvalues:
            assert self.dg is not None
            self.dg = self.tdc.broadcast(  # type: ignore
                self.dg,
                src=src,
                group=group,
            )
        else:
            assert self.dgda is not None
            self.dgda = self.tdc.broadcast(  # type: ignore
                self.dgda,
                src=src,
                group=group,
            )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        """Compute A inverse on assigned rank.

        update_a_factor() must be called at least once before this function.

        Args:
            damping (float, optional): damping value to condition inverse
                (default: 0.001).
        """

        if not isinstance(self.a_factor, torch.Tensor):
            raise RuntimeError(
                'Cannot eigendecompose A before A has been computed',
            )

        if self.symmetric_factors:
            try:
                self.da, self.qa = torch.linalg.eigh(
                    self.a_factor.to(torch.float32),
                )
            except Exception as e:
                print(f"eigen a symmetric decomposition error: {e} at {self.name}")
                if torch.isnan(self.a_factor).any() or torch.isinf(self.a_factor).any():
                    print(f"nan or inf in a_factor at {self.name} try to fix")
                    self.a_factor.nan_to_num_()
                try :
                    print(f"gpu memory usage at {self.name} before fix: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
                    print(f"try to fix a_factor at {self.name}")
                    epsilon = 0.007
                    matrix = self.a_factor + epsilon * torch.eye(self.a_factor.size(0), dtype=self.a_factor.dtype, device=self.a_factor.device)
                    matrix = (matrix + matrix.t()) / 2
                    self.da, self.qa = torch.linalg.eigh(
                        (matrix).to(torch.float32),
                    )
                except Exception as e:
                    print(f"eigen a decomposition error again: {e} at {self.name} ,pass")
        else:
            try:
                da, qa = torch.linalg.eig(
                    self.a_factor.to(torch.float32),
                )
            except Exception as e:
                print(f"eigen a non-symm decomposition error: {e} at {self.name}")
                if torch.isnan(self.a_factor).any() or torch.isinf(self.a_factor).any():
                    print(f"nan or inf in a_factor at {self.name} try to fix")
                    self.a_factor = torch.nan_to_num(self.a_factor)
                try:
                    print(f"try to fix a_factor at {self.name}")
                    epsilon = 0.007
                    matrix = self.a_factor + epsilon * torch.eye(self.a_factor.size(0), dtype=self.a_factor.dtype, device=self.a_factor.device)
                    matrix = (matrix + matrix.t()) / 2
                    da, qa = torch.linalg.eig(
                        (matrix).to(torch.float32),
                    )
                except Exception as e:
                    print(f"eigen a decomposition error: {e} at {self.name} ,pass")
            self.da = da.real
            self.qa = qa.real
        self.qa = cast(torch.Tensor, self.qa).to(self.inv_dtype)
        self.da = cast(torch.Tensor, self.da).to(self.inv_dtype)
        self.da = torch.clamp(self.da, min=0.0)

    def compute_g_inv(self, damping: float = 0.001) -> None:
        """See `compute_g_inv`."""

        if not isinstance(self.g_factor, torch.Tensor):
            raise RuntimeError(
                'Cannot eigendecompose G before G has been computed',
            )
        
        if self.symmetric_factors:
            try:
                self.dg, self.qg = torch.linalg.eigh(
                    self.g_factor.to(torch.float32),
                )
            except Exception as e:
                print(f"eigen g symmetric decomposition error: {e} at {self.name}")
                if torch.isnan(self.g_factor).any() or torch.isinf(self.g_factor).any():
                    print(f"nan or inf in g_factor at {self.name} try to fix")
                    self.g_factor = torch.nan_to_num(self.g_factor)
                try:
                    print(f"try to fix g_factor at {self.name}")
                    epsilon = 0.007
                    matrix = self.g_factor + epsilon * torch.eye(self.g_factor.size(0), dtype=self.g_factor.dtype, device=self.g_factor.device)
                    matrix = (matrix + matrix.t()) / 2
                    self.dg, self.qg = torch.linalg.eigh(
                        (matrix).to(torch.float32),
                    )
                except Exception as e:
                    print(f"eigen g decomposition error again: {e} at {self.name}")
                    raise e
        else:
            try :
                dg, qg = torch.linalg.eig(
                    self.g_factor.to(torch.float32),
                )
            except Exception as e:
                print(f"eigen g non-sym decomposition error: {e} at {self.name}")
                if torch.isnan(self.g_factor).any() or torch.isinf(self.g_factor).any():
                    print(f"nan or inf in g_factor at {self.name} try to fix")
                    self.g_factor = torch.nan_to_num(self.g_factor)
                try:
                    epsilon = 0.007
                    matrix = self.g_factor + epsilon * torch.eye(self.g_factor.size(0), dtype=self.g_factor.dtype, device=self.g_factor.device)
                    dg, qg = torch.linalg.eig(
                        (matrix).to(torch.float32),
                    )
                except Exception as e:
                    print(f"eigen g decomposition error: {e} at {self.name} ,pass")
            self.dg = dg.real
            self.qg = qg.real
        assert self.dg is not None
        #assert self.da is not None
        self.qg = cast(torch.Tensor, self.qg).to(self.inv_dtype)
        self.dg = self.dg.to(self.inv_dtype)
        self.dg = torch.clamp(self.dg, min=0.0)
        if self.prediv_eigenvalues:
            assert self.da is not None
            self.dgda = 1 / (torch.outer(self.dg, self.da) + damping)
            self.dg = None
            self.da = None

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        if (
            self.qa is None
            or self.qg is None
            or (not self.prediv_eigenvalues and self.da is None)
            or (not self.prediv_eigenvalues and self.dg is None)
            or (self.prediv_eigenvalues and self.dgda is None)
        ):
            raise RuntimeError(
                'Eigendecompositions for both A and G have not been computed',
            )
        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.qa.dtype)
        v1 = self.qg.t() @ grad @ self.qa
        if self.prediv_eigenvalues:
            v2 = v1 * self.dgda
        else:
            v2 = v1 / (
                torch.outer(
                    cast(torch.Tensor, self.dg),
                    cast(torch.Tensor, self.da),
                )
                + damping
            )
        self.grad = (self.qg @ v2 @ self.qa.t()).to(grad_type)
