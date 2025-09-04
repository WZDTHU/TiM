torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_dpgbench_ddp.py \
    --config configs/t2i/tim_xl_p1_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type dpgbench \
    --caption-dir projects/evaluate/dpg_bench/prompts \
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
    --config configs/t2i/tim_xl_p1_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type dpgbench \
    --caption-dir projects/evaluate/dpg_bench/prompts \
    --height 1024 \
    --width 1024 \
    --per-proc-batch-size 32 \
    --cfg-scale 2.5 \
    --num-steps 16 \
    --slice_vae 

