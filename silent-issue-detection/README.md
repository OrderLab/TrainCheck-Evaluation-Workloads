### 🐛 Detailed Silent Issue Bug Table

| **Bug ID**                  | **Failure Location** | **AE?** | **Trace Type**     | **Shared Trace With**     | **Invariant Source**      | **Repro Notes**                                            |
|----------------------------|----------------------|--------|--------------------|---------------------------|---------------------------|------------------------------------------------------------|
| `baichuan2-86`             | HW/Driver            | ✅ Yes | Real               | `pytorch-84803`           | Same as `pytorch-84803`   | Requires 2 GPUs                                             |
| `deepspeed-1801`           | Framework            | ✅ Yes | Real               | –                         | DS GPT Pretraining (clean) | Requires TP setup, modified HF Megatron-DS build           |
| `deepspeed-5794`           | Framework            | ❌ No  | –                  | –                         | –                         | Invariant relation under evaluation                        |
| `lightning-thunder-725`    | Framework            | ✅ Yes | Simulated          | –                         | Clean pipeline (autocast) | Simulated autocast issue; non-trivial Lightning setup      |
| `mmpretrain-702`           | Framework            | ✅ Yes | Simulated          | –                         | Clean mmpretrain          | Simulated trace due to complex setup                       |
| `pytorch-51800`            | Framework            | ✅ Yes | Real               | –                         | MNIST example             | Reproducible with any standard MNIST pipeline              |
| `pytorch-84803`            | HW/Driver            | ✅ Yes | Real               | Used by `baichuan2-86`, `96600` | Clean MNIST (X-DDP)    | Requires 2 GPUs, X-Dist/DPP setup                          |
| `pytorch-96600`            | HW/Driver            | ✅ Yes | Same as `84803`    | `pytorch-84803`           | Same as `pytorch-84803`   | See `pytorch-84803`                                        |
| `pytorch-104336`           | Framework            | ✅ Yes | Real               | Used by `x-jxmnop-ddp-out-of-sync` | Clean DS1801 or DDP | Requires multi-GPU with DDP or DS                          |
| `pytorch-115607`           | Compiler             | ✅ Yes | Simulated          | –                         | MNIST example             | Simulated issue due to codegen bug                         |
| `pytorch-forum-84911`      | User Code            | ✅ Yes | Real               | –                         | MNIST example             | Covered in 5-min tutorial; Adam misconfig                  |
| `stackoverflow-60335387`   | User Code            | ✅ Yes | Real               | –                         | MNIST example             | Same Adam misuse as `84911`                                |
| `stackoverflow-67180955`   | Framework            | ❌ No  | –                  | –                         | –                         | Requires outdated Python version                           |
| `transformers-17877`       | Framework            | ✅ Yes | Real               | –                         | Clean Transformers run     | Standard setup                                              |
| `transformers-23723`       | Framework            | ✅ Yes | Real               | –                         | Transformers (long input)  | Use `max_new_tokens` > input length                        |
| `transformers-33844`       | Framework            | ✅ Yes | Real               | –                         | tensor.normal_ w/ instr    | Requires instr descriptors enabled                         |
| `transformers-34204`       | Framework            | ❌ No  | –                  | –                         | –                         | Invariant support still in progress                        |
| `x-jxmnop-ddp-out-of-sync` | User Code            | ✅ Yes | Same as `104336`   | `pytorch-104336`          | Same as `104336`          | DDP desync bug; shares trace and invariant setup           |