torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_mjhq30k_ddp.py \
    --config configs/t2i/tim_xl_p1_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type mjhq30k \
    --caption-dir projects/evaluate/mjhq30k/meta_data.json \
    --height 1024 \
    --width 1024 \
    --per-proc-batch-size 32 \
    --cfg-scale 1.5 \
    --num-steps 4 \
    --slice_vae 


torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    projects/sample/sample_t2i_mjhq30k_ddp.py \
    --config configs/t2i/tim_xl_p1_t2i.yaml \
    --ckpt checkpoints/t2i_model.bin \
    --sample-dir ./samples \
    --data-type mjhq30k \
    --caption-dir projects/evaluate/mjhq30k/meta_data.json \
    --height 1024 \
    --width 1024 \
    --per-proc-batch-size 32 \
    --cfg-scale 1.5 \
    --num-steps 16 \
    --slice_vae 

