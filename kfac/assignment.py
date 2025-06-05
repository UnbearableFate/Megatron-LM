"""Work assignment interface and implementations."""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable
import numpy as np
import torch.distributed as dist


@dataclass(frozen=True)
class _Group:
    """Dataclass for tracking ranks and group handle together."""

    ranks: frozenset[int]
    group: Any


@dataclass(frozen=True)
class _LayerFactors:
    """Dataclass for tracking layer name and factors of the layer together."""

    layer: str
    factors: list[str]


class WorkAssignment(metaclass=ABCMeta):
    """Abstract Interface to a Work Assignment Class."""

    def __repr__(self) -> str:
        """String representation of the work assignment."""
        layer_strs = []
        for layer in self.get_layers():
            factors = self.get_factors(layer)
            invs = {
                factor: self.inv_worker(layer, factor) for factor in factors
            }
            layer_strs.append(
                f'  layer="{layer}": '
                f'is_grad_worker={self.is_grad_worker(layer)}, '
                f'src_grad_worker={self.src_grad_worker(layer)}, '
                f'inv_workers={invs}',
            )
        s = ',\n'.join(layer_strs)
        return f'{self.__class__.__name__}(\n{s}\n)'

    @abstractmethod
    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast."""
        raise NotImplementedError

    @abstractmethod
    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast."""
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> tuple[str, ...]:
        """Return tuple of layers assigned."""
        raise NotImplementedError

    @abstractmethod
    def get_factors(self, layer: str) -> tuple[str, ...]:
        """Return tuple of factors associated with the layer."""
        raise NotImplementedError

    @abstractmethod
    def inv_worker(self, layer: str, factor: str) -> int:
        """Return rank that computes inverse factor for this layer."""
        raise NotImplementedError

    @abstractmethod
    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer."""
        raise NotImplementedError

    @abstractmethod
    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        raise NotImplementedError

    @abstractmethod
    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors."""
        raise NotImplementedError

    @abstractmethod
    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        raise NotImplementedError


class KAISAAssignment(WorkAssignment):
    """Work assignment strategy implementation for KAISA."""

    def __init__(
        self,
        work: dict[str, dict[str, float]],
        *,
        local_rank: int,
        world_size: int,
        grad_worker_fraction: float,
        group_func: Callable[[list[int]], dist.ProcessGroup | None],
        colocate_factors: bool = True,
    ) -> None:
        """Init KAISAAssignment.

        Args:
            work (dict[str, dict[str, int]]): dictionary mapping unique layer
                names to sub-dictionaries where the keys are the str names for
                each factor associated with the layer and the values are the
                cost of each factor computation for load balancing.
            local_rank (int): local rank of process as assigned by the
                distributed backend.
            world_size (int): number of workers in the environment.
            grad_worker_fraction (float): fraction of the workers in the world
                that should be responsible for computing the gradient for a
                given layer. I.e. the gradient worker count is max(1,
                world_size * grad_worker_fraction).
            group_func (callable): callable for making communication process
                groups (e.g., torch.distributed.ProcessGroup). The callable
                should take an iterable of ranks in the group.
            colocate_factors (bool): if True, assign all factors for a layer to
                the same inverse worker. Otherwise, distribute the factors
                across layers in the gradient worker group (default: False).
        """
        self.work = work
        if 0 > grad_worker_fraction or 1 < grad_worker_fraction:
            raise ValueError(
                'grad_worker_fraction must be in [0, 1]. '
                f'Got {grad_worker_fraction}.',
            )
        if 0 > local_rank:
            raise ValueError('local_rank must be > 0')
        if 0 > world_size:
            raise ValueError('world_size must be > 0')
        grad_workers = max(1, world_size * grad_worker_fraction)
        if grad_workers != int(grad_workers):
            raise ValueError(
                'world_size*grad_worker_fraction must produce an integer '
                f'value. Found {world_size}*{grad_worker_fraction}'
                f'={grad_workers}.',
            )
        else:
            grad_workers = int(grad_workers)
        if local_rank >= world_size:
            raise ValueError(
                'local_rank={local_rank} larger than world_size={world_size}',
            )
        self.local_rank = local_rank
        self.world_size = world_size
        self.grad_worker_fraction = grad_worker_fraction
        self.grad_workers = grad_workers
        self.group_func = group_func
        self.colocate_factors = colocate_factors

        grad_worker_ranks = self.partition_grad_workers(
            self.world_size,
            self.grad_workers,
        )
        grad_receiver_ranks = self.partition_grad_receivers(
            self.world_size,
            self.grad_workers,
        )

        ranks_to_communication_group: dict[
            frozenset[int],
            dist.ProcessGroup | None,
        ] = {}
        for ranks in grad_worker_ranks | grad_receiver_ranks:
            # TODO(gpauloski): some group configurations resulted in
            #   dist.new_group returning the same handle for distinct
            #   rank groups
            ranks_to_communication_group[ranks] = self.group_func(list(ranks))

        if dist.get_rank() == 0:
            print(f"work: {work}")
        self._inv_assignments = self.greedy_assignment(
            work,
            [list(ranks) for ranks in grad_worker_ranks],
            #self.world_size,
            self.colocate_factors,
        )

        self._grad_receiver_groups: dict[str, _Group] = {}
        self._grad_worker_groups: dict[str, _Group] = {}
        for layer in self._inv_assignments:
            inv_worker = list(self._inv_assignments[layer].values()).pop()
            for ranks in grad_worker_ranks:
                if inv_worker in ranks:
                    self._grad_worker_groups[layer] = _Group(
                        ranks=ranks,
                        group=ranks_to_communication_group[ranks],
                    )
            for ranks in grad_receiver_ranks:
                if self.local_rank in ranks:
                    self._grad_receiver_groups[layer] = _Group(
                        ranks=ranks,
                        group=ranks_to_communication_group[ranks],
                    )

    @staticmethod
    def greedy_assignment(
        work: dict[str, dict[str, float]],
        worker_groups: list[list[int]],
        colocate_factors: bool,
    ) -> dict[str, dict[str, int]]:
        """Greedy constrained layer work assignments.

        Assigns work units to ranks in a lowest-current load greedy approach.

        Args:
            work: dict mapping layer names to a sub-dict that maps work for
                the layer (e.g., factors) to the approximate cost of that
                work object.
            worker_groups: list of list of ranks where each sub-list of ranks
                represents a worker group. All work (e.g., factor computations)
                for a given layer will be constrained to be workers within
                a worker group. For example, if the worker groups are
                [[0, 1], [2, 3]], there will never be a case where the two
                factors for a given layer are performed on worker in separate
                groups.
            world_size (int): world_size
            colocate_factors (bool): if true, factors for a single layer will
                be assigned to the same worker. Otherwise, factors for a single
                layer can be computed on separate workers given those workers
                are in the same group.

        Returns:
            dict matching the structure of the work inputs except the values
            of the sub-dicts are the worker ranks that the corresponding factor
            should be computed on.
        """
        #worker_loads = [0.0] * world_size
        worker_loads = {}
        for group in worker_groups:
            for i in group:
                worker_loads[i] = 0.0

        assignments = {
            layer: {factor: -1 for factor in factors}
            for layer, factors in work.items()
        }

        summed_work = {
            layer: sum(factors.values()) for layer, factors in work.items()
        }
        sorted_groups = [
            layer
            for layer, _ in sorted(
                summed_work.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        for layer in sorted_groups:
            # Sum up loads across workers in each worker group
            worker_group_loads = [
                sum(worker_loads[i] for i in group) for group in worker_groups
            ]
            # Get worker group with lowest current load
            worker_group = worker_groups[
                worker_group_loads.index(min(worker_group_loads))
            ]

            if colocate_factors:
                _worker_group_loads = [worker_loads[i] for i in worker_group]
                min_worker = worker_group[
                    _worker_group_loads.index(min(_worker_group_loads))
                ]
                worker_loads[min_worker] += summed_work[layer]
                for factor in work[layer]:
                    assignments[layer][factor] = min_worker
            else:
                # Sort items in the item group by descending cost
                factors = sorted(
                    work[layer].items(),
                    key=lambda x: (x[1], x[0]),
                    reverse=True,
                )
                # Perform lowest current load greedy assignment within worker
                # and layer factors group
                for factor, cost in factors:
                    _worker_group_loads = [
                        worker_loads[i] for i in worker_group
                    ]
                    min_worker = worker_group[
                        _worker_group_loads.index(min(_worker_group_loads))
                    ]
                    worker_loads[min_worker] += cost
                    assignments[layer][factor] = min_worker

        for layer in assignments:
            for factor in assignments[layer]:
                assert assignments[layer][factor] >= 0
        print("worker_loads: ", worker_loads)
        return assignments

    @staticmethod
    def greedy_assignment_efficiency_new(
        work: dict[str, dict[str, float]],
        workers: list[int],
        colocate_factors: bool,
        computational_efficiency: dict[int, float],
        ) -> dict[str, dict[str, int]]:
        '''
        Greedy layer-factor assignment with respect to computational efficiency.
        Now uses only one worker group (workers), and encourages
        larger tasks to go to higher-efficiency (faster) workers.

        Args:
            work: 
                A dict mapping layer names to a dict of {factor_name: cost}.
                e.g. work["layer1"] = {"factorA": 10.0, "factorB": 5.0}
            workers:
                A list of worker ranks, e.g. [0, 1, 2].
            colocate_factors:
                If True, all factors of a layer must be assigned to the same worker;
                otherwise, factors of the same layer can be assigned to different workers.
            computational_efficiency:
                A dict mapping worker rank -> computational efficiency (GFLOPs/s or any relative measure).
                e.g. {0: 1.0, 1: 2.0, 2: 1.5}.

        Returns:
            assignments:
                A dict of the same structure as `work`, but values replaced by the
                worker rank to which each factor is assigned. 
                e.g. assignments["layer1"]["factorA"] = 1
        '''

        # 1) 初始化每个 worker 的负载（表示当前已经累积的“时间”）
        worker_loads = {w: 0.0 for w in workers}

        # 2) 准备返回结果，结构同 work
        assignments = {
            layer: {factor: -1 for factor in factors} 
            for layer, factors in work.items()
        }

        # 3) 计算每个 layer 的总工作量，并按从大到小排序
        #    这样先分配大任务，可以更好地利用高效节点
        summed_work = {layer: sum(factors.values()) for layer, factors in work.items()}
        layers_sorted = sorted(summed_work, key=summed_work.get, reverse=True)

        # 4) 遍历层（由大到小）
        for layer in layers_sorted:
            # 当前层所有 factor 的工作量
            factors_dict = work[layer]
            
            if colocate_factors:
                # 4.1) 需要将整层的所有 factor 分配给同一个节点
                total_cost = summed_work[layer]

                # 找到分配此层后 "负载 + total_cost / efficiency" 最小的节点
                best_worker = None
                best_time = float('inf')
                for w in workers:
                    # 计算分配给 worker w 后，总的时间负载
                    time_if_assigned = worker_loads[w] + (total_cost / computational_efficiency[w])
                    if time_if_assigned < best_time:
                        best_time = time_if_assigned
                        best_worker = w

                # 更新选中节点的负载
                worker_loads[best_worker] = best_time

                # 将该 layer 的所有 factor 分配给同一个节点
                for factor in factors_dict:
                    assignments[layer][factor] = best_worker

            else:
                # 4.2) 同一层内的 factors 可以分开分配
                #      先对 factor 按照 cost 从大到小排序，优先分配大 factor
                sorted_factors = sorted(factors_dict.items(), key=lambda x: x[1], reverse=True)
                
                for factor_name, factor_cost in sorted_factors:
                    # 为每个 factor 找到 "分配后负载" 最小的 worker
                    best_worker = None
                    best_time = float('inf')
                    for w in workers:
                        time_if_assigned = worker_loads[w] + (factor_cost / computational_efficiency[w])
                        if time_if_assigned < best_time:
                            best_time = time_if_assigned
                            best_worker = w

                    # 更新负载并记录分配结果
                    worker_loads[best_worker] = best_time
                    assignments[layer][factor_name] = best_worker

        # 5) 确保所有 factor 都已分配给有效的 worker
        for layer in assignments:
            for factor in assignments[layer]:
                assert assignments[layer][factor] >= 0, "Factor assignment failed."
        print("worker_loads_new: ", worker_loads)
        return assignments

    @staticmethod
    def partition_grad_workers(
        world_size: int,
        grad_workers: int,
    ) -> set[frozenset[int]]:
        """Returns set of sets of unique gradient workers.

        Constructs an m x n grid of the ranks in the world where m=grad_workers
        and and n=world_size/grad_workers with ranks ordered in ascending
        order left-to-right, top-to-bottom. The gradient worker groups are the
        columns of this grid.

        Example:
            input: world_size = 8, grad_workers = 2

            |          grad_worker groups           |
            | group 1 | group 2 | group 3 | group 4 |
            | ------- | ------- | ------- | ------- |
            |    0    |    1    |    2    |    3    | <- grad receiver group 1
            |    4    |    5    |    6    |    7    | <- grad receiver group 2

            output: [[0, 4], [1, 5], [2, 6], [3, 7]]

        Args:
            world_size (int): world size.
            grad_workers (int): number of gradient workers.

        Returns:
            set[set[int]] where the total number of elements is equal to
            world_size and the size of each subset is equal to grad_workers.
        """
        if not 0 < world_size:
            raise ValueError('world_size must be > 0')
        if world_size % grad_workers != 0:
            raise ValueError(
                'world_size must be an integer multiple of the gradient '
                'worker count',
            )
        partitions = world_size // grad_workers
        return {
            frozenset(range(i, world_size, partitions))
            for i in range(partitions)
        }

    @staticmethod
    def partition_grad_receivers(
        world_size: int,
        grad_workers: int,
    ) -> set[frozenset[int]]:
        """Returns set of sets of unique gradient receiver groups.

        Constructs the grid described in `partition_grad_receivers` and returns
        the rows.

        Args:
            world_size (int): world size.
            grad_workers (int): number of gradient workers.

        Returns:
            set[set[int]] where the total number of elements is equal to
            world_size and the size of each top-level set is equal to
            grad_workers.
        """
        if not 0 < world_size:
            raise ValueError('world_size must be > 0')
        if world_size % grad_workers != 0:
            raise ValueError(
                'world_size must be an integer multiple of the gradient '
                'worker count',
            )
        partitions = world_size // grad_workers
        return {
            frozenset(range(i * partitions, i * partitions + partitions))
            for i in range(grad_workers)
        }

    def broadcast_gradients(self) -> bool:
        """Return if gradients need to be broadcast.

        In KAISA, this is True when the gradient worker count is less than
        world size (i.e., not the COMM-OPT case).
        """
        return self.grad_workers < self.world_size

    def broadcast_inverses(self) -> bool:
        """Return if inverses need to be broadcast.

        In KAISA, this is True when the gradient worker count is greater than
        1 (i.e., not the MEM-OPT case).
        """
        return self.grad_workers > 1

    def get_layers(self) -> tuple[str, ...]:
        """Return tuple of layers assigned."""
        return tuple(self._inv_assignments.keys())

    def get_factors(self, layer: str) -> tuple[str, ...]:
        """Return tuple of factors associated with the layer."""
        return tuple(self._inv_assignments[layer].keys())

    def inv_worker(self, layer: str, factor: str) -> int:
        """Return rank that computes inverse factor for this layer."""
        return self._inv_assignments[layer][factor]

    def is_grad_worker(self, layer: str) -> bool:
        """Return if this rank is a gradient worker for this layer."""
        return self.local_rank in self._grad_worker_groups[layer].ranks

    def src_grad_worker(self, layer: str) -> int:
        """Return rank that will share preconditioned gradient.

        If process is a gradient worker, this method should return the
        process rank. Otherwise, if the process is a gradient receiver, this
        method returns the rank that is responsible for sending the
        preconditioned gradient to this process.
        """
        return set(
            self._grad_worker_groups[layer].ranks
            & self._grad_receiver_groups[layer].ranks,
        ).pop()

    def factor_group(
        self,
        layer: str,
        factor: str,
    ) -> dist.ProcessGroup | None:
        """Communication group for allreducing factors.

        KAISA assumes strong data-parallel training, i.e., each rank in the
        world will contribute factors computed from its local mini-batch.
        Thus, this function simply returns the global process group.
        """
        return None

    def grad_worker_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for inverse factor broadcast.

        This communication group is used for the broadcasts of the inverses
        from the inverse worker to the remaining gradient workers for the
        layer.
        """
        return self._grad_worker_groups[layer].group

    def grad_receiver_group(self, layer: str) -> dist.ProcessGroup | None:
        """Return communication group for preconditioned gradient broadcast.

        This communication group is used for the broadcasts of the gradients
        from the gradient worker to the remaining gradient receivers for the
        layer.
        """
        return self._grad_receiver_groups[layer].group

if __name__ == '__main__':
    work = {'model.conv1': {'A': 19683, 'G': 262144}, 'model.layer1.0.conv1': {'A': 191102976, 'G': 262144}, 'model.layer1.0.conv2': {'A': 191102976, 'G': 262144}, 'model.layer1.1.conv1': {'A': 191102976, 'G': 262144}, 'model.layer1.1.conv2': {'A': 191102976, 'G': 262144}, 'model.layer1.2.conv1': {'A': 191102976, 'G': 262144}, 'model.layer1.2.conv2': {'A': 191102976, 'G': 262144}, 'model.layer2.0.conv1': {'A': 191102976, 'G': 2097152}, 'model.layer2.0.conv2': {'A': 1528823808, 'G': 2097152}, 'model.layer2.0.downsample.0': {'A': 262144, 'G': 2097152}, 'model.layer2.1.conv1': {'A': 1528823808, 'G': 2097152}, 'model.layer2.1.conv2': {'A': 1528823808, 'G': 2097152}, 'model.layer2.2.conv1': {'A': 1528823808, 'G': 2097152}, 'model.layer2.2.conv2': {'A': 1528823808, 'G': 2097152}, 'model.layer2.3.conv1': {'A': 1528823808, 'G': 2097152}, 'model.layer2.3.conv2': {'A': 1528823808, 'G': 2097152}, 'model.layer3.0.conv1': {'A': 1528823808, 'G': 16777216}, 'model.layer3.0.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer3.0.downsample.0': {'A': 2097152, 'G': 16777216}, 'model.layer3.1.conv1': {'A': 12230590464, 'G': 16777216}, 'model.layer3.1.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer3.2.conv1': {'A': 12230590464, 'G': 16777216}, 'model.layer3.2.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer3.3.conv1': {'A': 12230590464, 'G': 16777216}, 'model.layer3.3.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer3.4.conv1': {'A': 12230590464, 'G': 16777216}, 'model.layer3.4.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer3.5.conv1': {'A': 12230590464, 'G': 16777216}, 'model.layer3.5.conv2': {'A': 12230590464, 'G': 16777216}, 'model.layer4.0.conv1': {'A': 12230590464, 'G': 134217728}, 'model.layer4.0.conv2': {'A': 97844723712, 'G': 134217728}, 'model.layer4.0.downsample.0': {'A': 16777216, 'G': 134217728}, 'model.layer4.1.conv1': {'A': 97844723712, 'G': 134217728}, 'model.layer4.1.conv2': {'A': 97844723712, 'G': 134217728}, 'model.layer4.2.conv1': {'A': 97844723712, 'G': 134217728}, 'model.layer4.2.conv2': {'A': 97844723712, 'G': 134217728}, 'model.fc': {'A': 135005697, 'G': 1000}}
    world_size = 16
    assi = KAISAAssignment.greedy_assignment(work=work, worker_groups=[list(range(world_size))], colocate_factors=True)
    print(assi)
    print("====================================")
    assi2 = KAISAAssignment.greedy_assignment_efficiency_new(work=work, workers=list(range(world_size)), colocate_factors=True, computational_efficiency={i: 1.0 for i in range(world_size)})
    print(assi2)