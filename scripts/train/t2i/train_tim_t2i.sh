NNODES=2
GPUS_PER_NODE=8
MASTER_ADDR="localhost"
export MASTER_PORT=60563
mkdir -p workdir/t2i/tim_t2i
CMD=" \
    projects/train/trainer_t2i.py \
    --config configs/t2i/tim_xl_p1_t2i.yaml \
    --project_dir workdir/t2i/tim_t2i \
    --seed 0 \
    "
TORCHLAUNCHER="torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    "
bash -c "$TORCHLAUNCHER $CMD"