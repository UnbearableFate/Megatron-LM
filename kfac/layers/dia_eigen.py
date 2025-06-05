from .eigen import *

from typing import List
from kfac.gpt_mega.mini_eigen import MiniMLPEigen
from torch import Tensor

class DiaMLPEigen(KFACEigenLayer):
    """
    MiniEigen is a class that implements the KFAC algorithm for mini-batch training.
    It inherits from the KFACEigenLayer class and provides methods for computing
    the Fisher information matrix and its inverse using the mini-batch approach.
    """

    def __init__(
        self,
        module: ModuleHelper,
        name: str,
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
        """
        Initializes the MiniEigen class.

        Args:
            module (ModuleHelper): The module to be optimized.
            tdc (TorchDistributedCommunicator): The communicator for distributed training.
            allreduce_method (AllreduceMethod, optional): The method for all-reduce operation. Defaults to AllreduceMethod.ALLREDUCE.
            factor_dtype (torch.dtype, optional): The data type for factors. Defaults to None.
            grad_scaler (torch.cuda.amp.GradScaler | Callable[[], float] | None, optional): The gradient scaler. Defaults to None.
            inv_dtype (torch.dtype, optional): The data type for inverse computation. Defaults to torch.float32.
            symmetry_aware (bool, optional): Whether to use symmetry-aware optimization. Defaults to False.
            prediv_eigenvalues (bool, optional): Whether to pre-divide eigenvalues. Defaults to False.
        """
        super().__init__(
            module=module,
            tdc=tdc,
            allreduce_method=allreduce_method,
            factor_dtype=factor_dtype,
            grad_scaler=grad_scaler,
            inv_dtype=inv_dtype,
            symmetry_aware=symmetry_aware,
            prediv_eigenvalues=prediv_eigenvalues,
        )
        self.grp_size = 4

        device = self.module.module.weight.device
        dtype = self.inv_dtype
        # For A (input)
        self.qa_gathered : List[Tensor]= []
        self.da_gathered : List[Tensor]= []
        # For G (output)
        self.qg_gathered : List[Tensor]= []
        self.dg_gathered : List[Tensor]= []

        self.mini_eigens : List[MiniMLPEigen]= []
        for i in range(self.grp_size):
            
            mini_layer = MiniMLPEigen(
                module=module,
                name=name,
                tdc=tdc,
                allreduce_method=allreduce_method,
                factor_dtype=factor_dtype,
                grad_scaler=grad_scaler,
                inv_dtype=inv_dtype,
                symmetry_aware=symmetry_aware,
                prediv_eigenvalues=prediv_eigenvalues,
                group_size=self.grp_size,
                chunk_rank=i,
            )
            
            self.mini_eigens.append(mini_layer)
            self.qa_gathered.append(
                torch.empty(
                (mini_layer.a_factor_width, mini_layer.a_factor_width),
                dtype=dtype,
                device=device,
                )
            )
            self.da_gathered.append(
                torch.empty(
                (mini_layer.a_factor_width,),
                dtype=dtype,
                device=device,
                )
            )
            self.qg_gathered.append(
                torch.empty(
                (mini_layer.g_factor_width, mini_layer.g_factor_width),
                dtype=dtype,
                device=device,
                )
            )
            self.dg_gathered.append(
                torch.empty(
                (mini_layer.g_factor_width,),
                dtype=dtype,
                device=device,
                )
            )

        self.name = name
        self.in_features = module.module.weight.shape[1]  # only for Linear
        self.out_features = module.module.weight.shape[0]  # only for Linear
    
    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        """Save input for layer, but only use the slice corresponding to this rank."""

        for mini_eigen in self.mini_eigens:
            mini_eigen.save_layer_input(input_)

    def save_layer_grad_output(
        self,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Save grad_output for layer, but only use the slice corresponding to this rank."""
        for mini_eigen in self.mini_eigens:
            mini_eigen.save_layer_grad_output(grad_output)

    def update_a_factor(self, alpha: float = 0.95) -> None:
        """Update A factor using the mini-batch approach.

        Args:
            alpha (float, optional): The decay rate for the moving average. Defaults to 0.95.
        """
        for mini_eigen in self.mini_eigens:
            mini_eigen.update_a_factor(alpha=alpha)
    
    def update_g_factor(self, alpha: float = 0.95) -> None:
        """Update G factor using the mini-batch approach.

        Args:
            alpha (float, optional): The decay rate for the moving average. Defaults to 0.95.
        """
        for mini_eigen in self.mini_eigens:
            mini_eigen.update_g_factor(alpha=alpha)

    def compute_a_inv(self, damping: float = 0.001) -> None:
        for mini_eigen in self.mini_eigens:
            mini_eigen.compute_a_inv(damping=damping)
        # Gather each block's qa and da into self.qa_gathered and self.da_gathered
        for i, mini_eigen in enumerate(self.mini_eigens):
            assert self.qa_gathered[i].shape == mini_eigen.qa.shape
            assert self.da_gathered[i].shape == mini_eigen.da.shape
            self.qa_gathered[i] = mini_eigen.qa
            self.da_gathered[i] = mini_eigen.da
        
        self.qa = torch.block_diag(*self.qa_gathered)
        self.da = torch.cat(self.da_gathered)
    
    def compute_g_inv(self, damping: float = 0.001) -> None:
        for mini_eigen in self.mini_eigens:
            mini_eigen.compute_g_inv(damping=damping)
        # Gather each block's qg and dg into self.qg_gathered and self.dg_gathered
        for i, mini_eigen in enumerate(self.mini_eigens):
            assert self.qg_gathered[i].shape == mini_eigen.qg.shape
            assert self.dg_gathered[i].shape == mini_eigen.dg.shape
            self.qg_gathered[i] = mini_eigen.qg
            self.dg_gathered[i] = mini_eigen.dg
        
        self.qg = torch.block_diag(*self.qg_gathered)
        self.dg = torch.cat(self.dg_gathered)

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
        self.grad = grad.add_(self.qg @ v2 @ self.qa.t()).mul_(0.5).to(grad_type)
    