from kfac.layers.eigen import *
from kfac.layers.modules import get_cov, append_bias_ones

from typing import List
from enum import Enum

print_ct = {}


def print_once(message: str) -> None:
    """Print a message only once."""
    if message not in print_ct:
        print(message)
        print_ct[message] = 1
    else:
        print_ct[message] += 1


import torch.distributed as dist


class SplitEnd(Enum):
    IN = 0
    OUT = 1
    BOTH = 2
    NONE = 3


class MiniMLPEigenNew(KFACEigenLayer):
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
        split_end: SplitEnd = SplitEnd.BOTH,
        *,
        factor_dtype: torch.dtype | None = None,
        grad_scaler: torch.cuda.amp.GradScaler | Callable[[], float] | None = None,
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
            prediv_eigenvalues=False,
        )

        self.g_inv_gathered = None
        self.a_inv_gathered = None

        self.a_inv_local = None
        self.g_inv_local = None

        self.tp_group = tensor_parallel_dist_group
        chunk_rank = dist.get_rank(self.tp_group) if self.tp_group is not None else 0
        group_size = dist.get_world_size(self.tp_group) if self.tp_group is not None else 1
        self.name = name + f"_chunk{chunk_rank}"
        self.chunk_rank = chunk_rank
        self.grp_size = group_size

        self.split_in = True if split_end in [SplitEnd.IN, SplitEnd.BOTH] else False
        self.split_out = True if split_end in [SplitEnd.OUT, SplitEnd.BOTH] else False
        self.compute_chunk_setting(chunk_rank=chunk_rank, group_size=group_size)
        print(
            f"MiniEigen {self.name} initialized with split_in={self.split_in}, split_out={self.split_out}, in_out_size :{self.in_features}, {self.out_features}",
            f"a_factor_width={self.a_factor_width}, g_factor_width={self.g_factor_width}",
            f"moudule has bias: {self.module.has_bias()}, shape :{(self.module.module.bias.shape if self.module.has_bias() else None)}",
        )

    def compute_chunk_setting(self, chunk_rank: int, group_size: int = 2) -> None:
        """
        Compute the chunk start/end indices and allocate eigendecomposition buffers
        according to the split_end configuration.
        """
        in_features = self.module.module.weight.shape[1]
        out_features = self.module.module.weight.shape[0]
        self.in_features = in_features
        self.out_features = out_features
        device = self.module.module.weight.device
        dtype = self.inv_dtype

        if self.split_in:
            self._setup_input_chunk(chunk_rank, group_size, in_features, device, dtype)
            # assert self.in_features == self.a_factor_width * self.grp_size , \
            #     f"TODO: Input features {self.in_features} must be divisible by a_factor_width {self.a_factor_width}."
        else:
            self.input_chunk_start, self.input_chunk_end = 0, in_features
            self.a_factor_width = in_features + 1 if self.module.has_bias() else in_features

        if self.split_out:
            self._setup_output_chunk(chunk_rank, group_size, out_features, device, dtype)
            # assert self.out_features == self.g_factor_width * self.grp_size, \
            #     f"TODO: Output features {self.out_features} must be divisible by g_factor_width {self.g_factor_width}."
        else:
            self.output_chunk_start, self.output_chunk_end = 0, out_features
            self.g_factor_width = out_features

    def _setup_input_chunk(self, chunk_rank, group_size, in_features, device, dtype):
        self.input_chunk_start, self.input_chunk_end = self.get_chunk_start_end(
            chunk_rank, group_size, in_features
        )
        if chunk_rank == group_size - 1 and self.module.has_bias():
            self.a_factor_width = self.input_chunk_end - self.input_chunk_start + 1

        else:
            self.a_factor_width = self.input_chunk_end - self.input_chunk_start

        block_size = in_features // group_size

        if self.module.has_bias():
            self.a_inv_gathered = []
            for _ in range(group_size - 1):
                self.a_inv_gathered.append(
                    torch.empty((block_size, block_size), dtype=dtype, device=device)
                )
            self.a_inv_gathered.append(
                torch.empty((block_size + 1, block_size + 1), dtype=dtype, device=device)
            )
        else:
            self.a_inv_gathered = torch.empty(
                (in_features, self.a_factor_width), dtype=dtype, device=device
            )

    def _setup_output_chunk(self, chunk_rank, group_size, out_features, device, dtype):
        self.output_chunk_start, self.output_chunk_end = self.get_chunk_start_end(
            chunk_rank, group_size, out_features
        )
        self.g_factor_width = self.output_chunk_end - self.output_chunk_start

        self.g_inv_gathered = torch.empty(
            (out_features, self.g_factor_width), dtype=dtype, device=device
        )

    def get_chunk_start_end(self, chunk_rank, group_size, feature_num: int) -> tuple[int, int]:
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
        start = chunk_rank * chunk_size
        # 最后一个 rank 把多出来的维度也一起拿进去
        end = (chunk_rank + 1) * chunk_size if chunk_rank < group_size - 1 else feature_num
        return start, end

    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        if not self.split_in:
            super().save_layer_input(input_)
            return
        """Save input for layer, but only use the slice corresponding to this rank."""

        # 取出当前 rank 对应的子张量
        a = (
            input_[0][..., self.input_chunk_start : self.input_chunk_end]
            .to(self.factor_dtype)
            .clone()
        )

        # 计算局部的 A 因子
        def get_a_factor(a: torch.Tensor) -> torch.Tensor:
            """Compute A factor with the input from the forward pass.

            Args:
                a (torch.Tensor): tensor with shape batch_size * in_dim.
            """
            a = a.view(-1, a.size(-1))
            if self.module.has_bias() and self.chunk_rank == self.grp_size - 1:
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

    def save_layer_grad_output(self, grad_output: tuple[torch.Tensor, ...]) -> None:
        if not self.split_out:
            super().save_layer_grad_output(grad_output)
            return
        """Save grad w.r.t outputs for layer."""
        g = grad_output[0][..., self.output_chunk_start : self.output_chunk_end].to(
            self.factor_dtype
        )
        if self.grad_scaler is not None:
            g = g / self.grad_scaler()
        g = self.module.get_g_factor(g)
        if self._g_batch is None:
            self._g_batch = g
            self._g_count = 1
        else:
            self._g_batch = self._g_batch + g
            self._g_count += 1

    def all_gather_a_inv_tensors(self) -> None:
        """
        Gather per-rank computed qa and da tensors across tensor-parallel group
        and assemble into self.qa_gathered and self.da_gathered.
        """
        assert self.a_inv_gathered is not None, "a_inv_gathered must be initialized."
        assert self.a_inv_local is not None, "a_inv_local must be initialized."
        if self.module.has_bias():
            dist.all_gather(self.a_inv_gathered, self.a_inv_local, group=self.tp_group)
        else:
            dist.all_gather_into_tensor(self.a_inv_gathered, self.a_inv_local, group=self.tp_group)

    def all_gather_g_inv_tensors(self) -> None:
        """
        Gather per-rank computed qg and dg tensors across tensor-parallel group
        and assemble into self.qg_gathered and self.dg_gathered.
        """
        assert self.g_inv_gathered is not None, "g_inv_gathered must be initialized."
        assert self.g_inv_local is not None, "g_inv_local must be initialized."
        assert (
            self.g_inv_gathered.shape[0] == self.g_inv_local.shape[0] * self.grp_size
        ), f"g_inv_gathered.shape[0] {self.g_inv_gathered.shape[0]} must equal to g_inv_local.shape[0] {self.g_inv_local.shape[0]} * grp_size {self.grp_size}."

        dist.all_gather_into_tensor(self.g_inv_gathered, self.g_inv_local, group=self.tp_group)

    def compute_a_inv(self, damping: float = 0.001) -> None:
        super().compute_a_inv(damping=damping)
        # Gather each block's qa and da into self.qa_gathered and self.da_gathered
        inv_vals = 1.0 / (self.da + damping)  # (k,)
        F = self.qa.clone().mul_(inv_vals.unsqueeze(0))  # 先克隆，再对每列 in-place 缩放
        self.a_inv_local = torch.mm(F, self.qa.t())  # 整体乘一次
        self.qa = None
        self.da = None

        if self.split_in:
            self.all_gather_a_inv_tensors()

    def compute_g_inv(self, damping: float = 0.001) -> None:
        super().compute_g_inv(damping=damping)
        # Gather each block's qg and dg into self.qg_gathered and self.dg_gathered
        inv_vals_g = 1.0 / (self.dg + damping)  # dg 已经>0
        Qg_scaled = self.qg * inv_vals_g.unsqueeze(0)
        self.g_inv_local = Qg_scaled @ self.qg.t()
        self.qg = None
        self.dg = None

        if self.split_out:
            self.all_gather_g_inv_tensors()

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        if self.a_inv_local is None or self.g_inv_local is None:
            return

        '''
        def get_grad(modu) -> torch.Tensor:
            """Get formatted gradients (weight and bias) of module.

            Returns:
                gradient of shape If bias != None,
                concats bias.
            """
            g = cast(torch.Tensor, modu.module.weight.main_grad)
            if g is None:
                raise RuntimeError(
                    f"MiniEigen {self.name} get_grad, module weight grad is None, "
                    "make sure to call backward() before this method."
                )
            if modu.has_bias():
                g = torch.cat([g, modu.module.bias.main_grad.view(-1, 1)], 1)  # type: ignore
            return g
        '''

        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.a_inv_local.dtype)
        # self.grad = (self.g_inv @ grad @ self.a_inv).to(grad_type)
        if self.split_in:
            if self.split_out:
                # Block-diagonal left and right multiplication
                if not self.module.has_bias():
                    self.grad = block_diag_left_and_right(
                        A_blocks=self.g_inv_gathered,
                        B_blocks=grad,
                        C_blocks=self.a_inv_gathered,
                        is_a_transpose=False,
                        is_c_transpose=False,
                    )
                else:
                    self.grad = block_diag_right_mm_with_bias(
                        block_diag_left_mm(self.g_inv_gathered, grad), self.a_inv_gathered
                    )
            else:
                # Block-diagonal left multiplication
                self.grad = block_diag_right_mm(self.g_inv_local @ grad, self.a_inv_gathered)
                
        elif self.split_out:
            # Block-diagonal right multiplication
            self.grad = block_diag_left_mm(self.g_inv_gathered, grad) @ self.a_inv_local
            
        else:
            # Full matrix multiplication
            self.grad = self.g_inv_local @ grad @ self.a_inv_local
        # self.grad.mul_(0.5).add_(grad, alpha=0.5).to(dtype=grad_type)  # Add damping to the gradient
        self.grad.add_(grad).mul_(0.5).to(dtype=grad_type)  # Add damping to the gradient


def block_diag_left_and_right(
    A_blocks: torch.Tensor,
    B_blocks: torch.Tensor,
    C_blocks,
    is_a_transpose=False,
    is_c_transpose=False,
) -> torch.Tensor:
    """
    Multiply block-diagonal matrix (represented by stacked blocks) with block-structured matrix B.

    A_blocks: (kn, k) → n blocks of (k x k)
    B_blocks: (kn, km) → n×m blocks of (k x k)
    c_blocks: (km, k) → m blocks of (k x k)
    Returns: (kn, km)
    """
    kn, k = A_blocks.shape
    kn2, km = B_blocks.shape
    km2, k2 = C_blocks.shape
    assert kn == kn2, "Row count mismatch"
    assert kn % k == 0 and km % k == 0, "Dimensions must be divisible by k"
    assert km == km2 and k == k2, "C_blocks shape must match B_blocks"

    n = kn // k
    m = km // k

    # Reshape A to (n, k, k)
    A_reshaped = A_blocks.view(n, k, k)
    if is_a_transpose:
        A_reshaped = A_reshaped.transpose(1, 2)
    # Reshape B to (n, m, k, k)
    B_reshaped = B_blocks.view(n, k, m, k).permute(0, 2, 1, 3)  # (n, m, k, k)

    # Reshape C to (m, k, k)
    C_reshaped = C_blocks.view(m, k, k)
    if is_c_transpose:
        C_reshaped = C_reshaped.transpose(1, 2)

    # Block-wise multiplication
    D_blocks = torch.matmul(A_reshaped[:, None], B_reshaped)  # (n, m, k, k)
    D_blocks = torch.einsum("imjk,mkl->imjl", D_blocks, C_reshaped)

    # Reshape back to (kn, km)
    D = D_blocks.permute(0, 2, 1, 3).contiguous().reshape(n * k, m * k)
    return D


def block_diag_left_mm(A_blocks: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    kn, k = A_blocks.shape
    kn2, km_total = B_blocks.shape
    assert kn == kn2, "Row count mismatch"
    assert kn % k == 0, "kn must be divisible by k"
    n = kn // k

    if km_total % k == 0:
        m = km_total // k
        bias = False
    else:
        m = (km_total - 1) // k
        assert (km_total - 1) % k == 0, "Bias dimension must align with k"
        bias = True

    A_reshaped = A_blocks.view(n, k, k)
    B_main = B_blocks[:, : m * k].view(n, k, m, k).permute(0, 2, 1, 3)  # (n, m, k, k)
    C_main = torch.matmul(A_reshaped[:, None], B_main)  # (n, m, k, k)
    C_main = C_main.permute(0, 2, 1, 3).contiguous().reshape(n * k, m * k)

    if bias:
        # extract and reshape the last column (bias)
        B_bias = B_blocks[:, -1:].view(n, k, 1)  # (n, k, 1)
        C_bias = torch.matmul(A_reshaped, B_bias).reshape(n * k, 1)
        C = torch.cat([C_main, C_bias], dim=1)
    else:
        C = C_main
    return C


def block_diag_right_mm(
    B_flat: torch.Tensor, A_flat: torch.Tensor | List[torch.Tensor]
) -> torch.Tensor:
    if isinstance(A_flat, list):
        return block_diag_right_mm_with_bias(B_flat, A_flat)
    elif isinstance(A_flat, torch.Tensor):
        return block_diag_right_mm_no_bias(B_flat, A_flat)
    else:
        print_once(f"block_diag_right_mm: {A_flat}")
        raise TypeError("A_flat must be a torch.Tensor or a list of torch.Tensor")


def block_diag_right_mm_with_bias(B_flat: torch.Tensor, A_flat: List[torch.Tensor]) -> torch.Tensor:
    sizes = [a.shape[0] for a in A_flat]
    cum_sizes = [0]
    for s in sizes:
        cum_sizes.append(cum_sizes[-1] + s)
    C = B_flat.clone()
    for i in range(len(A_flat)):
        start = cum_sizes[i]
        end = cum_sizes[i + 1]
        C[:, start:end] = B_flat[:, start:end] @ A_flat[i]
    return C


def block_diag_right_mm_no_bias(
    B_flat: torch.Tensor, A_flat: torch.Tensor, is_a_transpose: bool = False
) -> torch.Tensor:
    """
    计算 C = B × A，其中 A 是带对角块的块对角矩阵，B 是一个按块分割的矩阵。

    参数:
    - B_flat: 形状 (n*k, m*k)，代表一个由 n×m 个 k×k 子块构成的大矩阵 B。
    - A_flat: 形状 (m*k, k)，代表一个由 m 个 k×k 对角块组成的块对角矩阵 A（按行堆叠）。
    - k:     每个块的大小 (k×k)。
    - is_a_transpose: 如果为 True，则在乘法前先对每个 k×k 的对角块做转置，
                      等价于 A_block = A_block.T。

    返回:
    - C: 形状 (n*k, m*k)，等于 B_flat × (block-diag(A_flat))，其中 block-diag(A_flat) 是一个 (m*k)×(m*k) 的对角块矩阵。
    """

    # --- 1. 检查 shape 约束 ---
    kn, km = B_flat.shape
    mk, k = A_flat.shape
    assert kn % k == 0 and km % k == 0 and mk % k == 0, "kn, km, mk 必须能被 k 整除"
    n = kn // k
    m = mk // k
    assert km == m * k, "B_flat 的列数应该等于 m*k"

    # --- 2. 将 B_flat reshape 成 (n, m, k, k) 格式的 sub-blocks ---
    # 首先把 B_flat 看作 (n*k, m*k)，reshape 为 (n, k, m, k)，再 permute 到 (n, m, k, k)
    B_reshaped = B_flat.view(n, k, m, k).permute(0, 2, 1, 3)
    # 此时 B_reshaped[i,j] 是一个 k×k 的子块，对应原来 B_flat 中行块 i、列块 j

    # --- 3. 将 A_flat reshape 成 (m, k, k) 格式的对角块 ---
    A_reshaped = A_flat.view(m, k, k)
    if is_a_transpose:
        # 如果需要先转置每个对角块
        A_reshaped = A_reshaped.transpose(1, 2)  # 变成 (m, k, k)，每个块内部转置

    # --- 4. 执行分块右乘：对每个 (i,j) 对做 B_ij @ A_j ---
    # B_reshaped 的 shape: (n, m, k, k)
    # A_reshaped 的 shape: (m, k, k)
    # 我们需要得到 C_blocks[i,j] = B_reshaped[i,j] @ A_reshaped[j]
    #
    # 方法 1：显式把 A_reshaped expand 成 (n, m, k, k)：
    #   A_expand = A_reshaped.unsqueeze(0).expand(n, m, k, k)
    #   C_blocks = torch.matmul(B_reshaped, A_expand)
    #
    # 方法 2：用 einsum 表达更直观：
    C_blocks = torch.einsum("imjk,mkl->imjl", B_reshaped, A_reshaped)
    # 解释：i: 0..n-1, m: 0..m-1, j: 0..k-1, k: 0..k-1, l: 0..k-1

    # 此时 C_blocks 形状为 (n, m, k, k)，其中 C_blocks[i,j] = B_reshaped[i,j] @ A_reshaped[j]

    # --- 5. 把 C_blocks 拼回 (n*k, m*k) 的格式 ---
    # 先 permute 到 (n, k, m, k)，再 reshape 成 (n*k, m*k)
    C = C_blocks.permute(0, 2, 1, 3).contiguous().view(n * k, m * k)
    return C
