mkdir ./results
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 bug_short.py
