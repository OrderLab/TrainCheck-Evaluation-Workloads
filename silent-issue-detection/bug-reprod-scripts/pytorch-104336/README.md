# PT-104336 

DistributedDataParallel Model Parameter Bug with CPU-GPU Transfers

Bug: https://github.com/pytorch/pytorch/issues/104336
Fix: Bad user behavior, currently no fix provided

## Overview

In `torch.nn.parallel.DistributedDataParallel` (DDP) training across multiple GPUs, moving model parameters between CPU and GPU can disrupt gradient synchronization. This happens because DDP registers autograd hooks to the original model parameters (backing tensors) during initialization, which are lost when moved to a different device. As a result, DDP will not synchronize gradients properly, leading to inconsistent model states across GPUs.

## The Bug

### Problem Summary
When a model wrapped in DDP is transferred from GPU to CPU and back to GPU, the DDP autograd hooks no longer reference the current parameters. Consequently, gradients do not synchronize correctly across processes, causing each GPU to compute its own gradients independently.

### Bug Reproduction Code

To expose this issue, the code moves the model to CPU and back to GPU:

```python
# Moving model to CPU and back to GPU
if move_model_to_cpu_and_back:
    ddp_model.cpu()
    ddp_model.cuda(rank)
```
This behavior assumes DDP will handle the new tensor references automatically. However, DDP cannot detect the parameter move and does not synchronize gradients, breaking model consistency.

## Fix Options
Two separate scripts provide solutions:

### Fix Option 1: Rewrap DDP after Device Move
This approach re-initializes DDP by deleting and rewrapping the model after moving it between devices. This ensures DDP registers its autograd hooks to the current parameters.

```python
# fix_1.py
if move_model_to_cpu_and_back:
    # Reinitialize DDP after moving between devices
    del ddp_model
    model.cpu()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
This ensures the new parameter references are correctly registered by DDP, enabling synchronization.
```
### Fix Option 2: Manual Offload and Restore of Parameter Data
This approach offloads the model parameters to CPU storage, frees their GPU storage, and then restores the data back to GPU. This avoids re-initializing DDP while maintaining the correct parameter references.

```python
# fix_2.py
from torch.distributed.utils import _alloc_storage, _free_storage

if move_model_to_cpu_and_back:
    # Manual parameter offloading and restoring
    with torch.no_grad():
        cpu_data = {}
        print("Offloading parameters to CPU storage")
        for name, param in model.named_parameters():
            cpu_data[name] = (param.data.cpu(), param.data.size())
            _free_storage(param.data)

        print("Reloading parameters back to GPU")
        for name, param in model.named_parameters():
            cpu_tensor, size = cpu_data[name]
            _alloc_storage(param.data, size)
            param.data.copy_(cpu_tensor)
```
This method keeps DDP intact while restoring the parameter data to GPU, preserving synchronization.

## Running the Fixes
To test both fixes, simply run:

```bash
# Test with Fix Option 1
python fix_1.py

# Test with Fix Option 2
python fix_2.py
```
Both options should now synchronize gradients correctly across all processes.
