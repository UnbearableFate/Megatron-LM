from .eigen import *

from typing import List
from kfac.gpt_mega.mini_eigen import MiniMLPEigen

class DiaMLPEigenBackup(KFACEigenLayer):
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
        self.grp_size = 2
        self.mini_eigens : List[MiniMLPEigen]= []
        for i in range(self.grp_size):
            self.mini_eigens.append(
                MiniMLPEigen(
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
            )
        self.qa_gathered= None
        self.qg_gathered = None
        self.da_gathered = None
        self.dg_gathered = None
        in_features = self.module.module.weight.shape[1]  # only for Linear
        out_features = self.module.module.weight.shape[0]

        in_block_size = in_features // self.grp_size
        out_block_size = out_features // self.grp_size

        device = self.module.module.weight.device
        dtype = self.inv_dtype
        # For A (input)
        self.qa_gathered = torch.empty(
            (in_block_size,in_features),
            dtype=dtype,
            device=device,
        )
        self.da_gathered = torch.empty(
            (in_features,),
            dtype=dtype,
            device=device,
        )
        # For G (output)
        self.qg_gathered = torch.empty(
            (out_block_size,out_features),
            dtype=dtype,
            device=device,
        )
        self.dg_gathered = torch.empty(
            (out_features,),
            dtype=dtype,
            device=device,
        )
        self.name = name
        
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
            print(f"{mini_eigen.name} updated A factor :shape {mini_eigen.a_factor.shape}")
    
    def update_g_factor(self, alpha: float = 0.95) -> None:
        """Update G factor using the mini-batch approach.

        Args:
            alpha (float, optional): The decay rate for the moving average. Defaults to 0.95.
        """
        for mini_eigen in self.mini_eigens:
            mini_eigen.update_g_factor(alpha=alpha)
            print(f"{mini_eigen.name} updated G factor :shape {mini_eigen.g_factor.shape}")

    def compute_a_inv(self, damping: float = 0.001) -> None:
        for mini_eigen in self.mini_eigens:
            mini_eigen.compute_a_inv(damping=damping)
        # Gather each block's qa and da into self.qa_gathered and self.da_gathered
        in_features = self.module.module.weight.shape[1]  # only for Linear
        in_block_size = in_features // self.grp_size
        for i, mini_eigen in enumerate(self.mini_eigens):
            start = i * in_block_size
            end = start + in_block_size
            self.qa_gathered[:,start:end] = mini_eigen.qa
            self.da_gathered[start:end] = mini_eigen.da
    
    def compute_g_inv(self, damping: float = 0.001) -> None:
        for mini_eigen in self.mini_eigens:
            mini_eigen.compute_g_inv(damping=damping)
        # Gather each block's qg and dg into self.qg_gathered and self.dg_gathered
        out_features = self.module.module.weight.shape[0]
        out_block_size = out_features // self.grp_size
        for i, mini_eigen in enumerate(self.mini_eigens):
            start = i * out_block_size
            end = start + out_block_size
            self.qg_gathered[:,start:end] = mini_eigen.qg
            self.dg_gathered[start:end] = mini_eigen.dg
    
    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute preconditioned gradient using block-diagonal eigen decomposition.

        Args:
            damping (float, optional): damping to use for eigen decomposition (default: 0.001).
        """
        if (
            self.qa_gathered is None
            or self.qg_gathered is None
            or self.da_gathered is None
            or self.dg_gathered is None
        ):
            raise RuntimeError(
                'Eigendecompositions for both A and G have not been computed',
            )

        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.qa_gathered.dtype)

        p = self.grp_size
        out_features, in_features = grad.shape
        out_chunk = out_features // p
        in_chunk = in_features // p

        # Reshape according to separate in/out block sizes
        qa_blocks = self.qa_gathered.view(p, in_chunk, in_chunk)
        da_blocks = self.da_gathered.view(p, in_chunk)
        qg_blocks = self.qg_gathered.view(p, out_chunk, out_chunk)
        dg_blocks = self.dg_gathered.view(p, out_chunk)

        # Partition grad into (p, p, out_chunk, in_chunk) and extract diagonal blocks
        grad_blocks = grad.view(p, out_chunk, p, in_chunk).permute(0, 2, 1, 3)

        new_grad_blocks = []
        for i in range(p):
            v1 = qg_blocks[i].t() @ grad_blocks[i, i] @ qa_blocks[i]
            denom = torch.outer(dg_blocks[i], da_blocks[i]) + damping
            v2 = v1 / denom
            grad_new_i = qg_blocks[i] @ v2 @ qa_blocks[i].t()
            new_grad_blocks.append(grad_new_i)

        self.grad = torch.block_diag(*new_grad_blocks).to(grad_type)
    
def block_diag_left_matmul(blocks_A: List[torch.Tensor], B: torch.Tensor) -> torch.Tensor:
    """
    Compute Y = A @ B where A = block_diag(blocks_A).
    
    - blocks_A: [A0, A1, ..., Ap-1], each Ai of shape (di, di).
    - B: Tensor of shape (D, M), where D = sum(di).
    
    Returns:
        Y of shape (D, M) given by:
        [A0 @ B0; A1 @ B1; ...; Ap-1 @ Bp-1],
        where Bi is the slice of B corresponding to rows of Ai.
    """
    # 1. Determine split sizes along rows of B
    split_sizes = [Ai.shape[0] for Ai in blocks_A]
    # 2. Split B into matching row slices
    B_slices = torch.split(B, split_sizes, dim=0)
    # 3. Multiply each block independently
    Y_slices = [Ai @ Bi for Ai, Bi in zip(blocks_A, B_slices)]
    # 4. Concatenate back into full result
    return torch.cat(Y_slices, dim=0)

def block_diag_right_matmul(A: torch.Tensor, blocks_B: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Y = A @ B where B = block_diag(blocks_B).
    
    - A: Tensor of shape (N, D), where D = sum(dj).
    - blocks_B: [B0, B1, ..., Bp-1], each Bj of shape (dj, dj).
    
    Returns:
        Y of shape (N, D) given by:
        [A0 @ B0, A1 @ B1, ..., Ap-1 @ Bp-1],
        where Aj is the slice of A corresponding to columns of Bj.
    """
    # 1. Determine split sizes along columns of A
    split_sizes = [Bj.shape[0] for Bj in blocks_B]
    # 2. Split A into matching column slices
    A_slices = torch.split(A, split_sizes, dim=1)
    # 3. Multiply each slice with its block
    Y_slices = [Ai @ Bj for Ai, Bj in zip(A_slices, blocks_B)]
    # 4. Concatenate back into full result
    return torch.cat(Y_slices, dim=1)


def block_diag_left_matmul_compact(A_compact, B, p):
    """
    Y = A @ B,  A = block_diag(A0,...,A_{p-1}),
    A_compact: (D, d) with D = p*d
    B:         (D, M)
    """
    D, d = A_compact.shape
    # 1) 把 A_compact 视为 (p, d, d)
    A3 = A_compact.view(p, d, d)
    # 2) 把 B 视为 (p, d, M)
    B3 = B.view(p, d, -1)
    # 3) 批量乘法
    Y3 = torch.matmul(A3, B3)        # (p, d, M)
    # 4) 重塑回 (D, M)
    return Y3.reshape(D, B.shape[1])

def block_diag_right_matmul_compact(A, B_compact, p):
    """
    Y = A @ B,  B = block_diag(B0,...,B_{p-1}),
    A:         (N, D) with D = p*d
    B_compact: (d, D)
    """
    N, D = A.shape
    d = D // p
    # 1) 把 B_compact 视为 (p, d, d)
    B3 = B_compact.view(d, p, d).permute(1, 0, 2)  # (p, d, d)
    # 2) 把 A 视为 (N, p, d)
    A3 = A.view(N, p, d)
    # 3) 分块相乘并水平拼回
    Y_slices = [A3[:, i, :] @ B3[i] for i in range(p)]  # 每块 (N, d)
    return torch.cat(Y_slices, dim=1)  # (N, D)