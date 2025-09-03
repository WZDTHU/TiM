import torch
import os
import re
import ast
import json
import pathlib
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as TF
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from torchvision.transforms import (
    Normalize,
    Resize,
    InterpolationMode,
    CenterCrop,
)
from einops import repeat
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List
from safetensors.torch import load_file, save_file
from PIL import Image
from tqdm import tqdm
    
# Image processing
CLIP_RESIZE = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
CLIP_NORMALIZE = Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
CENTER_CROP = CenterCrop(224)


def get_clip_fn(fn_type, device, precision="amp"):
    assert precision in ["bf16", "fp16", "amp", "fp32"], "Precision must be one of ['bf16', 'fp16', 'amp', 'fp32']"
    
    from transformers import CLIPProcessor, CLIPModel

    model_id = "checkpoints/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    model.requires_grad_(False).to(device)


    def extract_text_features(text_inputs: List[str], device: str) -> torch.Tensor:
        inputs = processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def extract_image_features(image_inputs: torch.Tensor, device: str) -> torch.Tensor:
        image_inputs = CLIP_RESIZE(image_inputs.to(device))
        image_inputs = CLIP_NORMALIZE(image_inputs)
        
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits: bool = False) -> torch.Tensor:
        model.to(image_inputs.device)
        
        inputs = processor(
            text=text_inputs,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # 将所有输入张量移动到设备
        inputs = {k: v.to(image_inputs.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            
            if return_logits:
                clip_score = outputs.logits_per_image.squeeze(1)
            else:
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                clip_score = (image_embeds * text_embeds).sum(-1)
                
        return clip_score

    
    if fn_type == 'text':
        return extract_text_features
    elif fn_type == 'image':
        return extract_image_features
    elif fn_type == 'score':
        return score_fn
    else:
        raise NotImplementedError(f"Function type '{fn_type}' is not supported.")
  

def get_openclip_fn(fn_type, device, precision="amp"):
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    import open_clip

    # model, preprocess = open_clip.create_model_from_pretrained("ViT-H-14", 'checkpoints/OpenCLIP--ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors')
    # tokenizer = open_clip.get_tokenizer("ViT-H-14", 'checkpoints/OpenCLIP--ViT-H-14-laion2B-s32B-b79K')
    model, preprocess = open_clip.create_model_from_pretrained("ViT-g-14", 'checkpoints/OpenCLIP-ViT-g-14-laion2B-s12B-b42K/open_clip_pytorch_model.safetensors')
    tokenizer = open_clip.get_tokenizer("ViT-g-14", 'checkpoints/OpenCLIP-ViT-g-14-laion2B-s12B-b42K')
    model.eval()
    model.requires_grad_(False).to(device)

    def extract_text_features(text_inputs, device):
        text_inputs = tokenizer(text_inputs, context_length=77).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)
        return text_features

    def extract_image_features(image_inputs, device):
        image_inputs = CLIP_RESIZE(image_inputs.to(device))
        image_inputs = CLIP_NORMALIZE(image_inputs)
        # embed
        image_features = model.encode_image(image_inputs, normalize=True)
        return image_features

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        text_inputs = tokenizer(text_inputs, context_length=77).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        clip_score = (image_features * text_features).sum(-1)
        if return_logits:
            clip_score = clip_score * model.logit_scale.exp()
        return clip_score
    
    if fn_type == 'text':
        return extract_text_features
    elif fn_type == 'image':
        return extract_image_features
    elif fn_type == 'score':
        return score_fn
    else:
        raise NotImplementedError

def get_longclip_fn(fn_type, device, precision="amp"):
    
    config = CLIPConfig.from_pretrained('checkpoints/LongCLIP-GmP-ViT-L-14')
    maxtokens = 248
    config.text_config.max_position_embeddings = maxtokens
    
    model = CLIPModel.from_pretrained('checkpoints/LongCLIP-GmP-ViT-L-14', config=config, attn_implementation="sdpa").eval()
    processor = CLIPProcessor.from_pretrained('checkpoints/LongCLIP-GmP-ViT-L-14')
    model.requires_grad_(False).to(device)
    if precision == "fp16":
        model.to(torch.float16)
    elif precision == "bf16":
        model.to(torch.bfloat16)

    def extract_text_features(text_inputs, device):
        with torch.no_grad():
            preprocessed = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=maxtokens,
                return_tensors="pt",
            ).to(device)

            text_embs = model.get_text_features(**preprocessed)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        return text_embs 
    
    def extract_image_features(image_inputs, device):
        pixel_values = CLIP_NORMALIZE(CENTER_CROP(CLIP_RESIZE(image_inputs.to(device))))

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        return image_embs

    def score_fn(image_inputs: torch.Tensor, text_inputs: str, return_logits=False):
        device = image_inputs.device
        model.to(device)

        pixel_values = CLIP_NORMALIZE(CENTER_CROP(CLIP_RESIZE(image_inputs)))

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.no_grad():
            preprocessed = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=maxtokens,
                return_tensors="pt",
            ).to(device)

            text_embs = model.get_text_features(**preprocessed)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
        # Get predicted scores from model(s)
        score = (text_embs * image_embs).sum(-1)
        
        if return_logits:
            score = score * model.logit_scale.exp()
        return score

    if fn_type == 'text':
        return extract_text_features
    elif fn_type == 'image':
        return extract_image_features
    elif fn_type == 'score':
        return score_fn
    else:
        raise NotImplementedError



def get_hpsv2_fn(fn_type, device, precision="amp"):
    precision = "amp" if precision == "no" else precision
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        "checkpoints/HPSv2/HPS_v2.1_compressed.pt",
        precision=precision,
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    tokenizer = get_tokenizer("ViT-H-14")
    
    model.eval()
    model.requires_grad_(False).to(device)

    def extract_text_features(text_inputs, device):
        text_inputs = tokenizer(text_inputs, context_length=77).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)
        return text_features

    def extract_image_features(image_inputs, device):
        for t in preprocess_val.transforms[2:]:
            image_inputs = torch.stack([t(img) for img in image_inputs])
        # embed
        image_features = model.encode_image(image_inputs, normalize=True)
        return image_features

    # gets vae decode as input
    def score_fn(
        image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False
    ):
        # Process pixels and multicrop
        model.to(image_inputs.device)
        for t in preprocess_val.transforms[2:]:
            image_inputs = torch.stack([t(img) for img in image_inputs])
        text_inputs = tokenizer(text_inputs, context_length=77).to(image_inputs.device)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)
            
        hps_score = (image_features * text_features).sum(-1)
    
        if return_logits:
            hps_score = hps_score * model.logit_scale.exp()
        return hps_score

    if fn_type == 'text':
        return extract_text_features
    elif fn_type == 'image':
        return extract_image_features
    elif fn_type == 'score':
        return score_fn
    else:
        raise NotImplementedError

def save_text_features(path, model_fn, batch_size, data_type, device):
    # with open(path[0], 'r') as fp:
    #     dataset = [ast.literal_eval(line) for line in fp]
    with open(path[0], 'r') as fp:
        dataset = json.load(fp=fp)
    run_iters = int(len(dataset) / batch_size)
    all_data = dict()
    if data_type == 'coco':
        for i in tqdm(range(run_iters)):
            names = [
                data['coco_url'].split('/')[-1].split('.')[0]
                for data in dataset[i*batch_size: (i+1)*batch_size]
            ]
            captions = [
                data['recaption'].encode('unicode-escape').decode('utf-8') 
                for data in dataset[i*batch_size: (i+1)*batch_size]
            ]
            text_features = model_fn(captions, device)
            for j in range(len(names)):
                all_data[names[j]] = text_features[j]
    elif data_type == 'mjhq':
        all_captions = []
        all_names = []
        for k, v in dataset.items():
            all_captions.append(v['prompt'])
            all_names.append(os.path.join(v['category'], k+'.jpg'))
        for i in tqdm(range(run_iters)):
            names = [name for name in all_names[i*batch_size: (i+1)*batch_size]]
            captions = [caption for caption in all_captions[i*batch_size: (i+1)*batch_size]]
            text_features = model_fn(captions, device)
            for j in range(len(names)):
                all_data[names[j]] = text_features[j]

    save_file(all_data, path[1])

def extract_base_name(input_string):
    # This pattern matches everything up to '_v' followed by digits, or the entire string if '_v' is not present
    match = re.match(r'(.*?)(?:_v\d+)?$', input_string)
    if match:
        base_name_regex = match.group(1)
    else:
        base_name_regex = input_string

    # Both methods should yield the same result, but we'll return the regex version for robustness
    return base_name_regex

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        path = pathlib.Path(root_dir)
        IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "JPEG"}
        self.files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.rglob("*.{}".format(ext))]
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        name = str(path).replace(self.root_dir, '')        
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, name
    
def calculate_clip(path, model_fn, batch_size, device):
    
    text_features = load_file(path[0])
    dataset = ImagePathDataset(path[1], transforms=TF.Compose([TF.ToTensor(), TF.Resize((224, 224))]))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
    )
    
    all_clip_score = []
    progress_bar = tqdm(range(0, len(dataloader)))
    for batch_id, batch in enumerate(dataloader):
        images, names = batch[0].to(device), batch[1]
        progress_bar.update(1)
        with torch.no_grad():
            image_features = model_fn(images, device)
        for i in range(len(names)):
            score = (text_features[names[i]].to(device) * image_features[i]).sum(-1)
            all_clip_score.append(score.cpu())
            
        # if batch_id % 10 == 0:
        #     print(torch.tensor(all_clip_score).mean().cpu() * 100)
    clip_score = torch.tensor(all_clip_score).mean() * 100
    return clip_score

    
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=500, help="Batch size to use")
parser.add_argument(
    "--device", type=str, default='cuda', help="Device to use. Like cuda, cuda:0 or cpu"
)
parser.add_argument(
    "--clip-type", type=str, default='clip'
)
parser.add_argument(
    "--data-type", type=str, default='coco'
)
parser.add_argument(
    "--save-stats",
    action="store_true",
    help=(
        "Generate an npz archive from a directory of "
        "samples. The first path is used as input and the "
        "second as output."
    ),
)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Paths to the generated images or " "to .npz statistic files"),
)

def main():
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.save_stats:
        fn_type = 'text'
    else:
        fn_type = 'image'
        # fn_type = 'score'
    
    if args.clip_type == 'clip':
        model_fn = get_clip_fn(fn_type, device)
    elif args.clip_type == 'openclip':
        model_fn = get_openclip_fn(fn_type, device)
    elif args.clip_type == 'hpsv2':
        model_fn = get_hpsv2_fn(fn_type, device)
    elif args.clip_type == 'longclip':
        model_fn = get_longclip_fn(fn_type, device)
    else:
        raise NotImplementedError
    
    if args.save_stats:
        save_text_features(args.path, model_fn, args.batch_size, args.data_type, device)
        return

    clip_score = calculate_clip(args.path, model_fn, args.batch_size, device)

    print(args.clip_type, 'evaluated: ', args.path[1])
    print("CLIP: ", clip_score)

if __name__ == '__main__':
    main()
    