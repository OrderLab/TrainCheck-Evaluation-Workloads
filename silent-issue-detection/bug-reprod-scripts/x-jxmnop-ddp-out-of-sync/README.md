# Performance degeneration w/o consistent DDP Wraping

 With the huggingface trainer, when training on multiple GPUs, sometimes the model is stored as an nn.Module, but sometimes it’s wrapped in this DistributedDataParallel thing 
- Issue: Inconsistent wrapping of the model in `DistributedDataParallel` (DDP).
- Context: Using Hugging Face Trainer for multi-GPU training. (we simulate this process by inheriting the `BuggyTrainer` from HF `Trainer`)
- Problem: Model wraps a new `DDP` for every new run, which does not synchronize loss and gradient, causing performance issues.
- Example: Conditional wrapping in `get_model_ddp_or_not` function.
- Result: Potential bugs and performance degradation.


Original Bug Source: 
https://x.com/jxmnop/status/1778520637193240892