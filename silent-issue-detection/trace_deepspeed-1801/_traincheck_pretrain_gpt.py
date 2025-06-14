
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/DS-1801-ML-DAIKON/Megatron-DeepSpeed/trace_ds-1801"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
from traincheck.instrumentor import VarSampler
"""Pretrain GPT"""
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices
from megatron.utils import average_losses_across_data_parallel_group
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os
try:
    from torch.distributed.elastic.multiprocessing.errors import record
    from traincheck.instrumentor.tracer import Instrumentor
    Instrumentor(record, scan_proxy_in_args=False, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
except ImportError:

    def record(fn):
        return fn
from traincheck.instrumentor import meta_vars
meta_vars['stage'] = 'init'

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    see_memory_usage(f'Before Building Model', force=True)
    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(), remote_device=None if args.remote_device == 'none' else args.remote_device, config_dict_or_path=args.deepspeed_config, enabled=args.zero_stage == 3, mpu=mpu):
        if args.deepspeed:
            attention_mask = torch.tril(torch.ones((1, args.seq_length, args.seq_length), device=torch.cuda.current_device())).view(1, 1, args.seq_length, args.seq_length)
            attention_mask = attention_mask < 0.5
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()
            args.attn_mask = attention_mask.to(torch.bool)
            model = GPTModelPipe(num_tokentypes=0, parallel_output=True)
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = GPTModel(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process)
    see_memory_usage(f'After Building Model', force=True)
    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    keys = ['text']
    datatype = torch.int64
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    (attention_mask, loss_mask, position_ids) = get_ltor_masks_and_position_ids(tokens, tokenizer.eod, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss, prefix_indices=None, loss_on_targets_only=args.loss_on_targets_only)
    return (tokens, labels, loss_mask, attention_mask, position_ids)

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()
    keys = ['text']
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    (attention_mask, loss_mask, position_ids) = get_ltor_masks_and_position_ids(tokens, tokenizer.eod, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss, prefix_indices=None, loss_on_targets_only=args.loss_on_targets_only)
    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()
    return ((tokens, position_ids, attention_mask), (labels, loss_mask))

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return (loss, {'lm loss': averaged_loss[0]})

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    timers('batch-generator').start()
    (tokens, labels, loss_mask, attention_mask, position_ids) = get_batch(data_iterator)
    timers('batch-generator').stop()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()
    return (output_tensor, partial(loss_func, loss_mask))

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    (train_ds, valid_ds, test_ds) = (None, None, None)
    print_rank_0('> building train, validation, and test datasets for GPT ...')
    if args.data_path:
        (train_ds, valid_ds, test_ds) = build_train_valid_test_datasets(data_prefix=args.data_path, data_impl=args.data_impl, splits_string=args.split, train_valid_test_num_samples=train_val_test_num_samples, seq_length=args.seq_length, seed=args.seed, skip_warmup=not args.mmap_warmup)
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append('train')
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append('valid')
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append('test')
        for s in assigned_train_valid_test:
            data_groups = zip(eval(f'args.{s}_weighted_split_paths'), eval(f'args.{s}_weighted_split_weights'), eval(f'args.{s}_weighted_split_splits'), eval(f'args.{s}_weighted_split_names'))
            for (paths, weights, splits, name) in data_groups:
                d = build_dataset_group(name, paths, weights, splits, args.data_impl, train_val_test_num_samples, args.seq_length, args.seed, not args.mmap_warmup, train_valid_test=s)
                eval(f'{s}_ds').append(d)
    else:
        raise NotImplementedError('No dataloading argument passed')
    print_rank_0('> finished creating GPT datasets ...')
    return (train_ds, valid_ds, test_ds)

@record
def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
if __name__ == '__main__':
    main()