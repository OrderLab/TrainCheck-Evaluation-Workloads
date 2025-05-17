**Original Issue Post:**
https://github.com/pytorch/pytorch/issues/84803

**Manifest:**
1. Dataloss when doing Data Parallel training on setups with multiple 4090 cards.

**Root Cause:**
1. Bad GPU P2P communication setup leading to dataloss during data transmission.

**Reproduction Script:**
`translation.py` and `run_translate.sh` reproduce the data loss bug in the training task of `facebook/m2m100_418M` model.

`translation.py` is adapted from [HugginggFace transformers' official pytorch example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py).
It modifies all PyTorch communication primitives to include a failure rate of **1%**. When the communication fails, the tensor being transmitted is zeroed out to mimic the data loss.

List of communication primitives modified:
- `torch.distributed.all_gather`
- `torch.distributed.all_reduce`
- `torch.distributed.send`
- `torch.distributed.recv`
- `torch.distributed.isend`
- `torch.distributed.irecv`
- `torch.Tensor.to`

### Dependencies:
```bash
transformers == 4.45.0
accelerate >= 0.12.0
datasets >= 1.8.0
sentencepiece != 0.1.92
protobuf
sacrebleu >= 1.4.12
py7zr
torch >= 1.3
evaluate
```

The bug reports (pytorch-84803 and pytorch-96600) are only reporting on the low-level individual bug level. While in Baichuan2-86, the issue reporter mentioned the e2e effect of the bug (e.g. non-sense output) but the effect was mentioned in the inference scenario. 



