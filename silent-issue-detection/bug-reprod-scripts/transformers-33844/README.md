# TF-33844
TF-33844: M2M100FlashAttention2 Dropout in Evaluation Mode

https://github.com/huggingface/transformers/pull/33844

## Bug Description 
The M2M100FlashAttention2 module does not disable dropout when the model is in evaluation mode (model.eval()), causing dropout to remain active during inference.

## Root Cause
The M2M100FlashAttention2 module does not automatically set the dropout rate to zero when self.training == False. As a result, the dropout mechanism is incorrectly applied during evaluation, introducing stochasticity in the outputs. This behavior likely originated from a shared implementation in the Bart model and has affected all models that copied this FlashAttention2 (FA2) implementation.

## Symptom (Silent)

Inconsistent Outputs: Dropout during inference leads to non-deterministic model outputs in evaluation mode, which is incorrect behavior and affects reproducibility.
Performance Degradation: Dropout unnecessarily reduces the model’s capacity during inference, potentially degrading performance on tasks where consistent output quality is critical.

## Code Exposing the Bug

Running model.eval() followed by repeated inference with the same input yields different outputs across runs, exposing that dropout is not disabled during evaluation.

## Fix

The PR fixes the issue by explicitly setting dropout = 0.0 when M2M100FlashAttention2.training == False, ensuring dropout is only applied during training and is skipped in evaluation mode, as expected.