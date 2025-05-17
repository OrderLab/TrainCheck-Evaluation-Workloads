## Manifest:
Dataloader workers applying the same augmentation to data, 
leading to identical data returned by different workers.

Reproducing the issue as per the script from: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

## TODO: work on an e2e example
https://github.com/descriptinc/melgan-neurips
https://github.com/facebookresearch/DepthContrast/issues/14

## Fix:
PR: https://github.com/pytorch/pytorch/pull/56488
Commit: aec83ff

## Dependencies:
The issue can be reproduced on torch 1.8.0
