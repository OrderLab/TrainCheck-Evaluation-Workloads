import os
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/TrainCheck-Evaluation-Workloads/silent-issue-detection/bug-reprod-scripts/pytorch-104336/traincheck_run_ddp-resnet18-cifar10_torch_2.2.2+cu121_2025-05-17_04-46-43"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
from traincheck.instrumentor import VarSampler
import torch.distributed as dist
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(dist, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.distributed
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch.distributed, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.nn.parallel import DistributedDataParallel as DDP
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DDP, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.multiprocessing as mp
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(mp, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn.functional as F
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(F, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader, DistributedSampler
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DataLoader, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DistributedSampler, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck import annotate_stage
from traincheck.instrumentor import meta_vars
annotate_stage('init')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
train_dset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_dset = datasets.CIFAR10('data', train=False, transform=transform)
model = models.resnet18(pretrained=False, num_classes=10).to(device)
model_sampler = VarSampler(model, var_name='model')

def setup(rank, world_size):
    """Sets up the process group and configuration for PyTorch Distributed Data Parallelism"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed environment"""
    dist.destroy_process_group()

def test(model, rank, epoch, test_loader):
    annotate_stage('testing')
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        batch_idx = 0
        for (data, target) in test_loader:
            (data, target) = (data.to(rank), target.to(rank))
            output = model(data)
            target = target.long()
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_idx += 1
            if batch_idx == 10:
                break
    test_loss /= len(test_loader.dataset)
    print(f'RANK: {rank} Epoch: {epoch} ' + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))

def _check_gradients_between_workers(world_size: int, move_model_to_cpu_and_back: bool, model: nn.Module, verbose: bool=False) -> None:
    """Gather gradients from all processes and check that they are equal."""
    for (k, v) in model.named_parameters():
        if v.grad is not None:
            grad_list = [torch.zeros_like(v.grad) for _ in range(world_size)]
            dist.all_gather(grad_list, v.grad, async_op=False)
            for grad in grad_list:
                if not torch.allclose(v.grad, grad):
                    if verbose:
                        print({'grad': v.grad.mean(), 'other_grad': grad.mean()})
                    print(f'move_model_to_cpu_and_back={move_model_to_cpu_and_back}. Gradients are not equal across processes (p={k})')

def _check_model_state_between_workers(world_size: int, model: nn.Module, verbose: bool=False) -> None:
    """Gather model states from all processes and check that they are equal."""
    for (k, v) in model.state_dict().items():
        if 'running_mean' in k or 'running_var' in k:
            continue
        state_list = [torch.zeros_like(v.data) for _ in range(world_size)]
        dist.all_gather(state_list, v.data, async_op=False)
        for state in state_list:
            if not torch.allclose(v.data, state):
                if verbose:
                    print({'state': v.mean(), 'other_state': state.mean()})
                print(f'Model states are not equal across processes (p={k})')

def train(rank, model, world_size):
    print(rank, type(model), type(world_size))
    setup(rank, world_size)
    annotate_stage('training')
    meta_vars['_DATA_PARALLEL_RANK'] = rank
    model = model.to(rank)
    model_sampler = VarSampler(model, var_name='model')
    print(f'Running DDP on rank {rank}.')
    train_sampler = DistributedSampler(train_dset)
    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=False, batch_size=64, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.cpu()
    ddp_model.cuda(rank)
    optimizer = optim.AdamW(ddp_model.parameters(), lr=0.001)
    model_sampler.register_hook(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        for (batch_idx, (data, target)) in enumerate(train_loader):
            (data, target) = (data.to(rank), target.to(rank))
            output = ddp_model(data)
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            dist.barrier()
            if batch_idx % 100 == 0:
                print(f'RANK: {rank} ' + 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))
            optimizer.zero_grad()
            if batch_idx == 10:
                break
        test(ddp_model, rank, epoch, test_loader)
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f'Running on {world_size} GPUs.')
    mp.spawn(train, args=(model, world_size), nprocs=world_size, join=True)
if __name__ == '__main__':
    main()