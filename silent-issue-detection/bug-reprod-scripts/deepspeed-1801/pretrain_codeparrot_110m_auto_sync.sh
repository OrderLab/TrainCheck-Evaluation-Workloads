GPUS_PER_NODE=8
TP_SIZE=4
PP_SIZE=1

CHECKPOINT_PATH=experiments/codeparrot-small-syncing
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=codeparrot_content_document
LAUNCHER="deepspeed --num_nodes 1 --num_gpus $GPUS_PER_NODE"
SCRIPT_PATH=/home/yuxuan/Megatron-DeepSpeed/pretrain_gpt.py

# TO USE THIS SCRIPT, YOU NEED TO ADD THE ARGUMENT 
# --layernorm-tp-auto-sync to argument.py in Megatron and reduce the layerNorm weights in each forward pass

# fused_layer_norm.py
# def forward(self, input):
# if self.layernorm_tp_auto_sync:
#     torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
#     torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
# 
# weights = [torch.empty_like(self.weight) for tp in range(mpu.get_tensor_model_parallel_world_size())]
# torch.distributed.all_gather(weights, self.weight, group=mpu.get_tensor_model_parallel_group())
# biases = [torch.empty_like(self.bias) for tp in range(mpu.get_tensor_model_parallel_world_size())]
# torch.distributed.all_gather(biases, self.bias, group=mpu.get_tensor_model_parallel_group())
# if any(torch.any(weight != self.weight) for weight in weights):
#     if mpu.get_tensor_model_parallel_rank() == 0:
#         print("Weight sync failed")
#         # print(weights)
# if any(torch.any(bias != self.bias) for bias in biases):
#     if mpu.get_tensor_model_parallel_rank() == 0:
#         print("Bias sync failed")
#         # print(biases)
# 
# return FusedLayerNormAffineFunction.apply(
#     input, self.weight, self.bias, self.normalized_shape,self.eps)

ARGS="
--tensor-model-parallel-size $TP_SIZE
--pipeline-model-parallel-size $PP_SIZE
--distributed-backend nccl

--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
--checkpoint-activations
--partition-activations
--exit-interval 20000

--merge-file $MERGE_FILE
--vocab-file $VOCAB_FILE

--save $CHECKPOINT_PATH
--load $CHECKPOINT_PATH
--data-path $DATA_PATH

--tensorboard-dir experiments/tensorboard-syncing
--tensorboard-queue-size 100
--log-timers-to-tensorboard
--log-batch-size-to-tensorboard
--log-validation-ppl-to-tensorboard

--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024

--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999

--clip-grad 1.0
--bf16

--log-level debug
--log-level-replica info

--deepspeed
--deepspeed-activation-checkpointing
--deepspeed_config experiments/bf16_config_codeparrot.json

--layernorm-tp-auto-sync
"

$LAUNCHER $SCRIPT_PATH $ARGS