torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_geneval_ddp.py \
    --config configs/t2i/tim_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type geneval \
    --caption-dir projects/evaluate/geneval/prompts/evaluation_metadata.jsonl \
    --height 1024 \
    --width 1024 \
    --per-proc-batch-size 32 \
    --cfg-scale 2.5 \
    --num-steps 4 \
    --slice_vae 


torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_geneval_ddp.py \
    --config configs/t2i/tim_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type geneval \
    --caption-dir projects/evaluate/geneval/prompts/evaluation_metadata.jsonl \
    --height 1024 \
    --width 1024 \
    --per-proc-batch-size 32 \
    --cfg-scale 2.5 \
    --num-steps 64 \
    --slice_vae 


torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_geneval_ddp.py \
    --config configs/t2i/tim_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type geneval \
    --caption-dir projects/evaluate/geneval/prompts/evaluation_metadata.jsonl \
    --height 2560 \
    --width 2560 \
    --per-proc-batch-size 4 \
    --cfg-scale 2.5 \
    --num-steps 4 \
    --slice_vae 