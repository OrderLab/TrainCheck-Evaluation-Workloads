# PyTorch-115607 Issue

https://github.com/pytorch/pytorch/issues/115607

## Environment Setup

1. Create a conda environment
2. Install PyTorch

```bash
conda create -n testenv python=3.11
conda activate testenv
conda install numpy=1.26.4
conda install pytorch=2.2.2 -c pytorch
```

## Run

```bash
python3 simple_buggy.py
python3 complex_buggy.py
```

## What to expect

`simple_buggy.py` and `complex_buggy.py` should have the same output for eager and dynamo mode.

## What is the bug

The training loop with `torch._dynamo.optimize("eager")` behaves differently compared to the default mode when gradients are set to `None`. In fact, all the parameter updates after the freezing iterations are all incorrect/frozen as expected.

In the `simple_buggy.py` example, the training loop increments the `step` counter correctly in the eager mode, whereas in the dynamo mode, the `step` counter fails to increment in the third iteration where gradients are `None`.
```
expected in eager:
step tensor(1.)
step tensor(1.)
step tensor(2.)
what actually happens after dynamo:
step tensor(1.)
step tensor(1.)
step tensor(1.)
```

The same discrepancy is observed in the `complex_buggy.py` example.
```
expected in eager:
step tensor(1.)
step tensor(2.)
step tensor(3.)
step tensor(3.)
step tensor(4.)
step tensor(5.)
what actually happens after dynamo:
step tensor(1.)
step tensor(2.)
step tensor(3.)
step tensor(3.)
step tensor(3.)
step tensor(3.)
```

## Root Causes

The issue might stem from how `torch._dynamo.optimize("eager")` interacts with the optimizer's state dictionary. Specifically, the optimizer's step counter may not increment correctly because the gradient accumulation and optimizer state updates behave differently in eager and dynamo modes. This discrepancy might be due to optimizations or transformations applied by `torch._dynamo` that affect the handling of gradient states.

## How to fix

1. Ensure that the gradient is not None before each call to optimizer.step()
2. Patch `torch._dynamo`: If torch._dynamo.optimize("eager") introduces problems, try verifying the problem without dynamic optimization.

## Potential Ways to Detect the Bug Automatically

1. Debug Logging: Add debug logging in `torch._dynamo` to trace the optimizer's state changes during training, making it easier to identify discrepancies. *Updates: maybe only detecting the `step` parameter can help.*
2. Invariant: Ensure that the gradients of all parameters are correctly calculated and updated before each call to optimizer.step(). In addition, make sure that the state of the parameters and the optimizer are also correctly updated after optimizer.step().
