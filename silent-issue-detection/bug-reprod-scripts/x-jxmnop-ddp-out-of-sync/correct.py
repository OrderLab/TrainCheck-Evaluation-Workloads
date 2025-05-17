import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12359"  # Replace with an available port

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def run(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    model = ToyModel().to(rank)

    if rank in [0,1,2,3]:
        print("Correct Behavior: Always wrap model in DDP")
        model = DDP(model, device_ids=[rank])  # Wrap only on rank 0 to mimic inconsistent wrapping
    else:
        print("Incorrect Behavior: Wrap model in DDP conditionally")
        model =  model  # Wrap only on rank 0 to mimic inconsistent wrapping
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # Sample forward pass
    x = torch.randn(10, 10).to(rank)
    output = model(x)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print loss on each GPU to observe desync
    print(f"Rank {rank} | Loss: {loss.item()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size)
