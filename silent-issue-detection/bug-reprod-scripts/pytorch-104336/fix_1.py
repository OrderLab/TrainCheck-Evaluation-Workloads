import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.utils import _alloc_storage, _free_storage


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.net1(x))
        return self.net2(x)


def ddp_check_grads_synchronized(move_model_to_cpu_and_back: bool) -> None:
    world_size = torch.cuda.device_count()
    mp.spawn(
        _ddp_check_grads_synchronized,
        args=(world_size, move_model_to_cpu_and_back),
        nprocs=world_size,
        join=True,
    )


def _ddp_check_grads_synchronized(rank: int, world_size: int, move_model_to_cpu_and_back: bool) -> None:
    print(f"Running DDP example on rank {rank}.")
    _ddp_setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    ddp_model = DDP(model, device_ids=[rank])

    if move_model_to_cpu_and_back:
        # Option 1: Rewrap model with DDP after moving it between CPU and GPU
        del ddp_model
        model.cpu()
        model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

    # create data
    x = torch.randn(20, 10)
    labels = torch.randn(20, 5)

    # forward pass
    x = x.to(rank)
    labels = labels.to(rank)
    outputs = ddp_model(x)

    # backward pass
    optimizer.zero_grad()
    loss = loss_fn(outputs, labels)
    loss.backward()

    _check_gradients_between_workers(world_size, move_model_to_cpu_and_back, ddp_model, verbose=rank == 0)
    _ddp_cleanup()


def _check_gradients_between_workers(
    world_size: int,
    move_model_to_cpu_and_back: bool,
    model: nn.Module,
    verbose: bool = False,
) -> None:
    """Gather gradients from all processes and check that they are equal."""
    for k, v in model.named_parameters():
        if v.grad is not None:
            grad_list = [torch.zeros_like(v.grad) for _ in range(world_size)]
            dist.all_gather(grad_list, v.grad, async_op=False)
            for grad in grad_list:
                if not torch.allclose(v.grad, grad):
                    if verbose:
                        print(
                            {
                                "grad": v.grad.mean(),
                                "other_grad": grad.mean(),
                            }
                        )
                    raise RuntimeError(
                        f"move_model_to_cpu_and_back={move_model_to_cpu_and_back}. "
                        f"Gradients are not equal across processes (p={k})"
                    )


def _ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12367"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _ddp_cleanup() -> None:
    dist.destroy_process_group()


if __name__ == "__main__":
    ddp_check_grads_synchronized(move_model_to_cpu_and_back=False)
    print("\nmove_model_to_cpu_and_back=False: SUCCESS!\n")

    ddp_check_grads_synchronized(move_model_to_cpu_and_back=True)
    print("\nmove_model_to_cpu_and_back=True: SUCCESS!\n")
