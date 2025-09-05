<h1 align="center">Transition Models: Rethinking the Generative Learning Objective</h1>



<div align="center">
  <a href="https://github.com/WZDTHU" target="_blank">ZiDong&nbsp;Wang</a><sup>1,2,*</sup> 
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://invictus717.github.io" target="_blank">Yiyuan&nbsp;Zhang</a><sup>1,2,*,‡</sup> 
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://yuexy.github.io/" target="_blank">Xiaoyu&nbsp;Yue</a><sup>2,3</sup> 
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://xyue.io" target="_blank">Xiangyu&nbsp;Yue</a><sup>1</sup> 
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://yg256li.github.io" target="_blank">Yangguang&nbsp;Li</a><sup>1,†</sup> 
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://wlouyang.github.io" target="_blank">Wanli&nbsp;Ouyang</a><sup>1,2</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="http://leibai.site" target="_blank">Lei&nbsp;Bai</a><sup>2,†</sup> 
  
  <sup>1</sup> MMLab CUHK &emsp; <sup>2</sup>Shanghai AI Lab &emsp; <sup>3</sup>USYD <br>
  <sup>*</sup>Equal Contribution &emsp; <sup>‡</sup>Project Lead &emsp; <sup>†</sup>Corresponding Authors &emsp; <br>
</div>



<h3 align="center">
<!-- [<a href="https://wzdthu.github.io/NiT">project page</a>]&emsp; -->
[<a href="https://arxiv.org/abs/2509.04394">arXiv</a>]&emsp;
[<a href="https://huggingface.co/GoodEnough/TiM-T2I">Model</a>]&emsp;
[<a href="https://huggingface.co/datasets/GoodEnough/TiM-Toy-T2I-Dataset">Dataset</a>]&emsp;
</h3>
<br>

<b>Highlights</b>: We propose Transition Models (TiM), a novel generative model that learns to navigate the entire generative trajectory with unprecedented flexibility. 
* Our Transition Models (TiM) are trained to master arbitrary state-to-state transitions. This approach allows TiM to learn the entire solution manifold of the generative process, unifying the few-step and many-step regimes within a single, powerful model. 
  ![Figure](./assets/illustration.png)
* Despite having only 865M parameters, TiM achieves state-of-the-art performance, surpassing leading models such as SD3.5 (8B parameters) and FLUX.1 (12B parameters) across all evaluated step counts on GenEval benchmark.Importantly, unlike previous few-step generators, TiM demonstrates monotonic quality improvement as the sampling budget increases. 
  ![Figure](./assets/nfe_demo.png)
* Additionally, when employing our native-resolution strategy, TiM delivers exceptional fidelity at resolutions up to $4096\times4096$.
  ![Figure](./assets/tim_demo.png)


## 🚨 News

- `2025-9-5` We are delighted to introduce TiM, which is the first text-to-image generator support any-step generation, entirely trained from scratch. We have released the codes and pretrained models of TiM.



## 1. Setup

First, clone the repo:
```bash
git clone https://github.com/WZDTHU/TiM.git && cd TiM
```

### 1.1 Environment Setup

```bash
conda create -n tim_env python=3.10
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn
pip install -r requirements.txt
pip install -e .
```


### 1.2 Model Zoo (WIP)


#### Text-to-Image Generation

A single TiM model can perform any-step generation (one-step, few-step, and multi-step) and demonstrate monotonic quality improvement as the sampling budget increases. 
| Model | Model Zoo | Model Size | VAE | 1-NFE GenEval | 8-NFE GenEval | 128-NFE GenEval |
|---------------|------------|---------|------------|-------|-------|-------|
| TiM-T2I | [🤗 HF](https://huggingface.co/GoodEnough/TiM-T2I/blob/main/t2i_model.bin) | 865M | [DC-AE](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) | 0.67 | 0.76 | 0.83 |



```bash
mkdir checkpoints
wget -c "https://huggingface.co/GoodEnough/TiM-T2I/resolve/main/t2i_model.bin" -O checkpoints/t2i_model.bin
```


#### Class-guided Image Generation:

| Model | Model Zoo | Model Size | VAE | 2-NFE FID | 500-NFE FID |
|---------------|------------|---------|------------|------------|------------|
| TiM-C2I-256 | [🤗 HF](https://huggingface.co/GoodEnough/TiM-C2I/blob/main/c2i_model_256.safetensors) | 664M | [SD-VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema) | 6.14 | 1.65  
| TiM-C2I-512 | [🤗 HF](https://huggingface.co/GoodEnough/TiM-C2I/blob/main/c2i_model_512.safetensors) | 664M | [DC-AE](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) | 4.79 | 1.69 


```bash
mkdir checkpoints
wget -c "https://huggingface.co/GoodEnough/TiM-C2I/resolve/main/c2i_model_256.safetensors" -O checkpoints/c2i_model_256.safetensors
wget -c "https://huggingface.co/GoodEnough/TiM-C2I/resolve/main/c2i_model_512.safetensors" -O checkpoints/c2i_model_512.safetensors
```


## 2. Sampling 

#### Text-to-Image Generation

We provide the sampling scripts on three benchmarks: GenEval, DPGBench, and MJHQ30K. You can specify the sampling steps, resolutions, and CFG scale in the corresponding scripts.

Sampling with TiM-T2I model on GenEval benchmark:
```bash
bash scripts/sample/t2i/sample_t2i_geneval.sh
```

Sampling with TiM-T2I model on DPGBench benchmark:
```bash
bash scripts/sample/t2i/sample_t2i_dpgbench.sh
```

Sampling with TiM-T2I model on MJHQ30k benchmark:
```bash
bash scripts/sample/t2i/sample_t2i_mjhq30k.sh
```

#### Class-guided Image Generation

We provide the sampling scripts for ImageNet-256 and ImageNet-512.

Sampling with C2I model on $256\times256$ resolution:
```bash
bash scripts/sample/c2i/sample_256x256.sh
```

Sampling with C2I model on $512\times512$ resolution:
```bash
bash scripts/sample/c2i/sample_512x512.sh
```


## 3. Evaluation


### Text-to-Image Generation

#### GenEval

Please follow the [GenEval](https://github.com/djghosh13/geneval) to setup the conda-environment. 

Given the directory of the generated images `SAMPLING_DIR` and folder of object dector `OBJECT_DETECTOR_FOLDER`, run the following codes:
```bash
python projects/evaluate/geneval/evaluation/evaluate_images.py $SAMPLING_DIR --outfile geneval_results.jsonl --model-path $OBJECT_DETECTOR_FOLDER
```
This will result in a JSONL file with each line corresponding to an image. Run the following codes to obtain the GenEval Score:
```bash
python projects/evaluate/geneval/evaluation/summary_scores.py geneval_results.jsonl
```


#### DPGBench
Please follow the [DPGBench](https://github.com/TencentQQGYLab/ELLA) to setup the conda-environment. 
Given the directory of the generated images `SAMPLING_DIR` , run the following codes:
```bash
python projects/evaluate/dpg_bench/compute_dpg_bench.py --image-root-path $SAMPLING_DIR --res-path dpgbench_results.txt --pic-num 4 
```

#### MJHQ30K
Please download [MJHQ30K](https://huggingface.co/datasets/playgroundai/MJHQ-30K) as the reference-image.


Given the directory of the reference-image direcotry `REFERENCE_DIR` and the directory of the generated images `SAMPLING_DIR`, run the following codes to calculate the FID Score:
```bash
python projects/evaluate/mjhq30k/calculate_fid.py $REFERENCE_DIR $SAMPLING_DIR
```

For CLIP Score, first compute the text features and save it in `MJHQ30K_TEXT_FEAT`:
```bash 
python projects/evaluate/mjhq30k/calculate_clip.py projects/evaluate/mjhq30k/meta_data.json $MJHQ30K_TEXT_FEAT/clip_feat.safetensors --save-stats
```
Then run the following codes to calculate the CLIP Score:
```bash
python projects/evaluate/mjhq30k/calculate_clip.py $MJHQ30K_TEXT_FEAT/clip_feat.safetensors $SAMPLING_DIR
```



### Class-guided Image Generation

The sampling generates a folder of samples to compute FID, Inception Score and other metrics. 
<b>Note that we do not pack the generate samples as a `.npz` file, this does not affect the calculation of FID and other metrics.</b>
Please follow the [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
to setup the conda-environment and download the reference batch. 

```bash
wget -c "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb" -O checkpoints/classify_image_graph_def.pb
```


Given the directory of the reference batch `REFERENCE_DIR` and the directory of the generated images `SAMPLING_DIR`, run the following codes:
```bash
python projects/evaluate/adm_evaluator.py $REFERENCE_DIR $SAMPLING_DIR
```





## 4. Training

### 4.1 Dataset Setup

Currently, we provide all the [preprocessed dataset](https://huggingface.co/datasets/GoodEnough/NiT-Preprocessed-ImageNet1K) for ImageNet1K. Please use the following commands to download the preprocessed latents.

```bash
bash tools/download_imagenet_256x256.sh
bash tools/download_imagenet_512x512.sh
```

For text-to-image generation, we provide a [toy dataset](https://huggingface.co/datasets/GoodEnough/TiM-Toy-T2I-Dataset). Please use the following command to download this dataset.
```bash
bash tools/download_toy_t2i_dataset.sh
```


### 4.2 Download Image Encoder

We use RADIO-v2.5-b as our image encoder for REPA-loss.

```bash
wget -c "https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-b_half.pth.tar" -O checkpoints/radio-v2.5-b_half.pth.tar
```


### 4.3 Training Scripts

Specify the `image_dir` in `configs/c2i/tim_b_p4.yaml` and train the base-model (131M) on ImageNet-256:
```bash
bash scripts/train/c2i/train_tim_c2i_b.sh
```

Specify the `image_dir` in `configs/c2i/tim_xl_p2_256.yaml` and train the XL-model (664M) on ImageNet-256:
```bash
bash scripts/train/c2i/train_tim_c2i_xl_256.sh
```

Specify the `image_dir` in `configs/c2i/tim_xl_p2_512.yaml` and train the XL-model (664M) on ImageNet-512:
```bash
bash scripts/train/c2i/train_tim_c2i_xl_512.sh
```

Specify the `root_dir` in `configs/t2i/tim_xl_p1_t2i.yaml` and train the T2I-model (865M) on Toy-T2I-Dataset:
```bash
bash scripts/train/t2i/train_tim_t2i.sh
```




## Citations
If you find the project useful, please kindly cite: 
```bibtex
@article{wang2025transition,
  title={Transition Models: Rethinking the Generative Learning Objective}, 
  author={Wang, Zidong and Zhang, Yiyuan and Yue, Xiaoyu and Yue, Xiangyu and Li, Yangguang and Ouyang, Wanli and Bai, Lei},
  year={2025},
  eprint={2509.04394},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
https://arxiv.org/abs/
## License
This project is licensed under the Apache-2.0 license.
