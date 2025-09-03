NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR="localhost"
export MASTER_PORT=60563
mkdir -p workdir/c2i/tim_xl_p2_256_mg
CMD=" \
    projects/train/trainer_c2i.py \
    --config configs/c2i/tim_xl_p2_256_mg.yaml \
    --project_dir workdir/c2i/tim_xl_p2_256_mg \
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