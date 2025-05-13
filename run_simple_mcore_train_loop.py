import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer
import torch.distributed as dist
import argparse
from datetime import timedelta

_SEQUENCE_LENGTH = 64

def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, timestamp=None):
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))

    backend = 'nccl' 
    timeout = timedelta(seconds=30)

    if timestamp is None:
        raise ValueError("timestamp must be provided")

    # 你固定的共享路径（NFS、Lustre）
    SHARE_FILES_DIR = "/work/NBB/yu_mingzhe/Megatron-LM/share_files"
    init_file = f"file://{SHARE_FILES_DIR}/pg_share{timestamp}"

    dist.init_process_group(
        backend=backend,
        init_method=init_file,
        rank=rank,
        world_size=world_size,
        timeout=timeout
    )

    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Initialized rank {rank}/{world_size} on device {torch.cuda.current_device()}")

    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=12, 
        num_attention_heads=4, 
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32,
    )

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=100, 
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model

def get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator

def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, required=True)
    args = parser.parse_args()

    initialize_distributed(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        timestamp=args.timestamp
    )

    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(20):
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False)

        optim.step()

        print(f'Losses reduced :  {losses_reduced}')

    # Saving the model
    ckpt_path = os.getcwd() + '/ckpt'
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # Loading the model
    gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    gpt_model.to(device)
    print('Successfully loaded the model')

