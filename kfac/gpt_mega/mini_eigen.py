from kfac.layers.eigen import *
from kfac.layers.modules import get_cov, append_bias_ones

from typing import List
from enum import Enum

class SplitEnd(Enum):
    IN = 0
    OUT = 1
    BOTH = 2
    NONE = 3

class MiniMLPEigen(KFACEigenLayer):
    """
    MiniEigen is a class that implements the KFAC algorithm for mini-batch training.
    It inherits from the KFACEigenLayer class and provides methods for computing
    the Fisher information matrix and its inverse using the mini-batch approach.
    """

    def __init__(
        self,
        module: ModuleHelper,
        name: str,
        tensor_parallel_dist_group: dist.ProcessGroup | None = None,
        #group_size : int = 2,
        #chunk_rank: int = -1,
        split_end: SplitEnd = SplitEnd.BOTH,
        *,
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
            tdc=None,
            factor_dtype=factor_dtype,
            grad_scaler=grad_scaler,
            inv_dtype=inv_dtype,
            symmetry_aware=symmetry_aware,
            prediv_eigenvalues=prediv_eigenvalues,
        )

        self.qa_gathered= None
        self.qg_gathered = None
        self.da_gathered = None
        self.dg_gathered = None
        self.sub_dist_group = None

        self.tp_group = tensor_parallel_dist_group
        chunk_rank = dist.get_rank(self.tp_group) if self.tp_group is not None else 0
        group_size = dist.get_world_size(self.tp_group) if self.tp_group is not None else 1
        
        self.name = name+f"_chunk{chunk_rank}"
        self.chunk_rank = chunk_rank
        self.grp_size = group_size
        self.split_in = True if split_end in [SplitEnd.IN, SplitEnd.BOTH] else False
        self.split_out = True if split_end in [SplitEnd.OUT, SplitEnd.BOTH] else False
        self.compute_chunk_setting(chunk_rank=chunk_rank, group_size=group_size)
        
    
    def compute_chunk_setting(
        self,
        chunk_rank: int,
        group_size: int = 2,
    ) -> None:
        """
        Compute the chunk start/end indices and allocate eigendecomposition buffers 
        according to the split_end configuration.
        """
        in_features = self.module.module.weight.shape[1]
        out_features = self.module.module.weight.shape[0]
        device = self.module.module.weight.device
        dtype = self.inv_dtype

        if self.split_in:
            self._setup_input_chunk(chunk_rank, group_size, in_features, device, dtype)
        else:
            self.input_chunk_start, self.input_chunk_end = 0, in_features
            self.a_factor_width = in_features + 1 if self.module.has_bias() else in_features
            self.qa_gathered = None
            self.da_gathered = None

        if self.split_out:
            self._setup_output_chunk(chunk_rank, group_size, out_features, device, dtype)
        else:
            self.output_chunk_start, self.output_chunk_end = 0, out_features
            self.g_factor_width = out_features
            self.qg_gathered = None
            self.dg_gathered = None

    def _setup_input_chunk(self, chunk_rank, group_size, in_features, device, dtype):
        self.input_chunk_start, self.input_chunk_end = self.get_chunk_start_end(
            chunk_rank, group_size, in_features
        )
        if chunk_rank == group_size - 1 and self.module.has_bias():
            self.a_factor_width = self.input_chunk_end - self.input_chunk_start + 1
        else:
            self.a_factor_width = self.input_chunk_end - self.input_chunk_start

        self.qa_gathered = torch.empty(
            (self.input_chunk_end - self.input_chunk_start, in_features),
            dtype=dtype,
            device=device,
        )
        self.da_gathered = torch.empty(
            (in_features,),
            dtype=dtype,
            device=device,
        )
    
    def _setup_output_chunk(self, chunk_rank, group_size, out_features, device, dtype):
        self.output_chunk_start, self.output_chunk_end = self.get_chunk_start_end(
            chunk_rank, group_size, out_features
        )
        self.g_factor_width = self.output_chunk_end - self.output_chunk_start

        self.qg_gathered = torch.empty(
            (self.output_chunk_end - self.output_chunk_start, out_features),
            dtype=dtype,
            device=device,
        )
        self.dg_gathered = torch.empty(
            (out_features,),
            dtype=dtype,
            device=device,
        )

    def get_chunk_start_end(self, chunk_rank, group_size ,feature_num: int) -> tuple[int, int]:
        """Get the start and end indices for the current rank's chunk of features."""
        """
        compute the start and end index of the chunk for the current rank
        Args:
            chunk_rank (int): The rank of the current chunk.
            group_size (int): The total number of chunks.
            feature_num (int): The total number of features.
        Returns:
            tuple[int, int]: The start and end indices of the chunk.
        """
        # 计算每个 rank 分得的特征宽度
        chunk_size = feature_num // group_size
        start = chunk_rank* chunk_size
        # 最后一个 rank 把多出来的维度也一起拿进去
        end = (chunk_rank+ 1) * chunk_size if chunk_rank < group_size - 1 else feature_num
        return start, end
    
    def create_sub_dist_group(self,rank_list) -> None:
        """Create a sub-distributed group for the current rank."""
        if self.sub_dist_group is None:
            self.sub_dist_group = torch.distributed.new_group(
                rank_list=rank_list)
        
    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        if not self.split_in:
            super().save_layer_input(input_)
            return
        """Save input for layer, but only use the slice corresponding to this rank."""

        # 取出当前 rank 对应的子张量
        a = input_[0][...,self.input_chunk_start:self.input_chunk_end].to(self.factor_dtype).clone()  

        # 计算局部的 A 因子
        def get_a_factor(a: torch.Tensor) -> torch.Tensor:
            """Compute A factor with the input from the forward pass.

            Args:
                a (torch.Tensor): tensor with shape batch_size * in_dim.
            """
            a = a.view(-1, a.size(-1))
            if self.module.has_bias() and self.chunk_rank == self.grp_size-1:
                a = append_bias_ones(a)
            return get_cov(a)
        
        a = get_a_factor(a)

        # 累加到全局统计里
        if self._a_batch is None:
            self._a_batch = a
            self._a_count = 1
        else:
            self._a_batch = self._a_batch + a
            self._a_count += 1 

    def save_layer_grad_output(
        self,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        if not self.split_out:
            super().save_layer_grad_output(grad_output)
            return
        """Save grad w.r.t outputs for layer."""
        g = grad_output[0][..., self.output_chunk_start:self.output_chunk_end].to(self.factor_dtype)
        if self.grad_scaler is not None:
            g = g / self.grad_scaler()
        g = self.module.get_g_factor(g)
        if self._g_batch is None:
            self._g_batch = g
            self._g_count = 1
        else:
            self._g_batch = self._g_batch + g
            self._g_count += 1

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
        chunk = grad.shape[0] // p

        qa_blocks = self.qa_gathered.view(p, chunk, -1)
        qg_blocks = self.qg_gathered.view(p, chunk, -1)
        da_blocks = self.da_gathered.view(p, chunk)
        dg_blocks = self.dg_gathered.view(p, chunk)
        
        grad_blocks = grad.view(p, chunk, p, chunk).permute(0, 2, 1, 3)

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