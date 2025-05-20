GPUS_PER_NODE=2
TP_SIZE=2
PP_SIZE=1

CHECKPOINT_PATH=experiments/codeparrot-small-short
VOCAB_FILE=/data/yuxuan/dataset/codeparrot/vocab.json
MERGE_FILE=/data/yuxuan/dataset/codeparrot/merges.txt
DATA_PATH=/data/yuxuan/dataset/codeparrot/codeparrot_content_document
LAUNCHER="deepspeed --num_nodes 1 --num_gpus $GPUS_PER_NODE"
SCRIPT_PATH=/home/yuxuan/DS-1801-ML-DAIKON/Megatron-DeepSpeed/_traincheck_pretrain_gpt.py


ARGS="
--tensor-model-parallel-size $TP_SIZE
--pipeline-model-parallel-size $PP_SIZE
--distributed-backend nccl

--log-interval 10
--save-interval 40
--eval-interval 20
--eval-iters 20
--checkpoint-activations
--partition-activations
--exit-interval 5

--merge-file $MERGE_FILE
--vocab-file $VOCAB_FILE

--save $CHECKPOINT_PATH
--load $CHECKPOINT_PATH
--data-path $DATA_PATH

--tensorboard-dir experiments/tensorboard
--tensorboard-queue-size 100
--log-timers-to-tensorboard
--log-batch-size-to-tensorboard
--log-validation-ppl-to-tensorboard

--num-layers 4
--hidden-size 64
--num-attention-heads 8
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
--deepspeed_config /home/yuxuan/DS-1801-ML-DAIKON/Megatron-DeepSpeed/experiments/bf16_config_codeparrot.json
"

$LAUNCHER $SCRIPT_PATH $ARGS
