import torch
import csv
import json
import os
import random
import ast
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from .sampler_utils import get_train_sampler, get_packed_batch_sampler



def resize_arr(pil_image, height, width):
    pil_image = pil_image.resize((width, height), resample=Image.Resampling.BICUBIC)

    return pil_image


class T2IDatasetMS(Dataset):
    def __init__(self, root_dir, packed_json, jsonl_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset = []
        with open(packed_json, 'r') as fp:
            self.packed_dataset = json.load(fp)
        
        with open(jsonl_dir, 'r') as fp:
            self.dataset = [json.loads(line) for line in fp]
        

    def __len__(self):
        return len(self.dataset)
    
    def get_one_data(self, data_meta):        
        data_item = dict()
        image_file = os.path.join(self.root_dir, data_meta['image_file'])

        image = Image.open(image_file).convert("RGB")
        
        bucket = data_meta['bucket']
        resolutions = bucket.split('-')[-1].split('x')
        height, width = int(int(resolutions[0])/32)*32, int(int(resolutions[1])/32)*32
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: resize_arr(pil_image, height, width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        image = transform(image)
        
        data_item['image'] = image
        data_item['caption'] = random.choice(data_meta['captions']).encode('unicode-escape').decode('utf-8')     

        return data_item

    def __getitem__(self, index):
        data_meta = self.dataset[index]
        # data_item = self.get_one_data(data_meta)
        try:
            data_item = self.get_one_data(data_meta)
        except:
            print(f"Warning: {data_meta['image_file']} does not exist", flush=True)
            data_item = None

        return data_item



def bucket_collate_fn(batch):
    caption = []
    image = []
    for data in batch:
        if data == None:
            continue
        caption.append(data['caption'])
        image.append(data['image'])
    image = torch.stack(image)
    return dict(image=image, caption=caption)




class T2ILoader():
    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config.dataloader.batch_size
        self.num_workers = data_config.dataloader.num_workers
        
        self.data_type = data_config.data_type

        if self.data_type == 'image_ms':
            self.train_dataset = T2IDatasetMS(**OmegaConf.to_container(data_config.dataset))
        else:
            raise
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, rank, world_size, global_batch_size, max_steps, resume_steps, seed):
        batch_sampler = get_packed_batch_sampler(
            self.train_dataset.packed_dataset, rank, world_size, max_steps, resume_steps, seed
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=bucket_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return None




