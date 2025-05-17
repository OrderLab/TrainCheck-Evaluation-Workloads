# TF-23723

Bug: https://github.com/huggingface/transformers/issues/23723
Fix: https://github.com/huggingface/transformers/pull/23724

## Symptom

- Latent: triggers the bug on the 13th iterations of the loop.
When specific long prompts are combined with certain audio inputs, an error occurs during inference. This issue is less predictable but appears linked to certain prompt and audio combinations.

## Menifestation

max_new_tokens Limitation Issue: When a very long prompt_ids sequence is provided, max_new_tokens does not correctly limit the number of generated tokens. For example, setting max_new_tokens=10 may still result in over 10 new tokens being generated (in the example, 25 tokens).

## Root Cause

The root cause seems to be that when prompt_ids are provided, max_new_tokens is recalculated using the length of text_prompt_ids before they are trimmed to fit the context window.

Therefore, the model fails to determine when to stop, causing errors during indexing if the prompt is too long or complex for the given context.

## Invariants

Invariant 1: Ensure max_new_tokens strictly limits the length of generated tokens. The final number of generated tokens should be less than or equal to max_new_tokens. (VarConsistency)
Invariant 2: Verify that prompt_ids are trimmed to fit within the model’s context length before forward API (APIVarConsistency)