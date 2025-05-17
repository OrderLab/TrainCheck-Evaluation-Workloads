# DeepSpeed 5794

Original Issue Post: https://github.com/microsoft/DeepSpeed/issues/5794

## Manifest
Training/Inference of MOE models get stuck mysteriously.

## Root Cause
DeepSpeed does not have proper handling for heterogenity across moe workers.
Inconsistently distributed tokens will cause inconsistency in invocation to `all_to_all`

## Dependency
The bug can be reproduced on deepspeed 0.15.4

## Invariant
APIArgs: `all_to_all` across workers in a single iteration should be invoked with certain args being consistent.