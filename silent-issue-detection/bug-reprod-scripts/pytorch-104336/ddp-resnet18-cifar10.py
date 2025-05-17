import os
import torch.distributed as dist
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader, DistributedSampler

# set seed to minimize randomness
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_dset = datasets.CIFAR10('data', train=False, transform=transform)




model = models.resnet18(pretrained=False, num_classes=10).to(device)
# model = BasicNet().to(device)

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

def test(model, rank, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            target = target.long()
            loss = criterion(output, target)
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f"RANK: {rank} Epoch: {epoch} " + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
                    print(
                        f"move_model_to_cpu_and_back={move_model_to_cpu_and_back}. "
                        f"Gradients are not equal across processes (p={k})"
                    )
                
def _check_model_state_between_workers(
    world_size: int,
    model: nn.Module,
    verbose: bool = False,
) -> None:
    """Gather model states from all processes and check that they are equal."""
    for k, v in model.state_dict().items():
        if "running_mean" in k or "running_var" in k:
            continue
        state_list = [torch.zeros_like(v.data) for _ in range(world_size)]
        dist.all_gather(state_list, v.data, async_op=False)
        for state in state_list:
            if not torch.allclose(v.data, state):
                if verbose:
                    print(
                        {
                            "state": v.mean(),
                            "other_state": state.mean(),
                        }
                    )
                print(
                    f"Model states are not equal across processes (p={k})"
                )

def train(rank, model, world_size):
    print(rank, type(model), type(world_size))
    setup(rank, world_size)
    model = model.to(rank)
    print(f"Running DDP on rank {rank}.")
    train_sampler = DistributedSampler(train_dset)
    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=False, batch_size=64, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.cpu()
    ddp_model.cuda(rank)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    # Train for one epoch
    for epoch in range(5):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            output = ddp_model(data)
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            dist.barrier()

            if batch_idx % 100 == 0:
                print(f"RANK: {rank} " + 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                _check_gradients_between_workers(
                    world_size,
                    True,
                    ddp_model,
                    verbose=rank == 0,
                )
                _check_model_state_between_workers(
                    world_size,
                    ddp_model,
                    verbose=rank == 0,
                )
            optimizer.zero_grad()
        # do test after each epoch
        test(ddp_model, rank, epoch, test_loader)
        
    cleanup()




def main():
    world_size = torch.cuda.device_count()
    # get rank
    print(f"Running on {world_size} GPUs.")
    mp.spawn(train, args=(model, world_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()