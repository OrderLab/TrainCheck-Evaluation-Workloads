import torch
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/ml-daikon-input-programs/pytorch/ddp-multigpu/traincheck_run_multigpu_torch_2.2.2+cu121_2025-03-09_15-31-19"

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
from traincheck.instrumentor import VarSampler
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn.functional as F
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(F, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.utils.data import Dataset, DataLoader
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(Dataset, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DataLoader, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from datautils import MyTrainDataset
import torch.multiprocessing as mp
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(mp, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.utils.data.distributed import DistributedSampler
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DistributedSampler, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.nn.parallel import DistributedDataParallel as DDP
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DDP, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.distributed import init_process_group, destroy_process_group
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(init_process_group, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(destroy_process_group, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import os
from traincheck import annotate_stage
from traincheck.instrumentor import meta_vars
annotate_stage('init')

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

class Trainer:

    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.BCEWithLogitsLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f'[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}')
        self.train_data.sampler.set_epoch(epoch)
        for (batch_idx, (source, targets)) in enumerate(self.train_data):
            meta_vars['step'] = len(self.train_data) * epoch + batch_idx
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = 'checkpoint.pt'
        torch.save(ckp, PATH)
        print(f'Epoch {epoch} | Training checkpoint saved at {PATH}')

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

class ToyModel(torch.nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = torch.nn.Linear(20, 15)
        self.net2 = torch.nn.Linear(15, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.net2(x)
        return x

def load_train_objs():
    train_set = MyTrainDataset(256)
    model = ToyModel()
    model_sampler = VarSampler(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model_sampler.register_hook(optimizer)
    return (train_set, model, optimizer)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    annotate_stage('init')
    ddp_setup(rank, world_size)
    meta_vars['_DATA_PARALLEL_RANK'] = rank
    (dataset, model, optimizer) = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    annotate_stage('training')
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)