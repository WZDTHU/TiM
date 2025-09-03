torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_c2i_ddp.py \
    --config configs/c2i/tim_xl_p1_512.yaml \
    --ckpt checkpoints/c2i_model_256.safetensors \
    --sample-dir ./samples \
    --height 512 \
    --width 512 \
    --per-proc-batch-size 32 \
    --T-max 1.0 \
    --cfg-scale 1.0 \
    --num-steps 1 \


torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_c2i_ddp.py \
    --config configs/c2i/tim_xl_p1_512.yaml \
    --ckpt checkpoints/c2i_model_256.safetensors \
    --sample-dir ./samples \
    --height 512 \
    --width 512 \
    --per-proc-batch-size 32 \
    --T-max 1.0 \
    --cfg-scale 3.0 \
    --guidance-low 0.0 \
    --guidance-high 0.7 \
    --stochasticity-ratio 0.2 \
    --num-steps 250