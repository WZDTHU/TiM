import os
import json
import datetime
import torchvision
import numpy as np
import torch

from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import hflip
from accelerate.logging import get_logger
from safetensors.torch import load_file
from .sampler_utils import get_train_sampler


logger = get_logger(__name__, log_level="INFO")


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class ImagenetDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return {"image": x, "label": y}

    def __len__(self):
        return len(self.dataset)

class ImagenetLatentDataset(Dataset):
    def __init__(self, latent_dir, image_dir, image_size):
        super().__init__()
        self.RandomHorizontalFlipProb = 0.5
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.Lambda(lambda pil_image: (pil_image, hflip(pil_image))),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])        
        
        self.dataset = []
        for class_folder in os.listdir(image_dir):
            if os.path.isfile(os.path.join(image_dir, class_folder)):
                continue
            latent_class_folder = os.path.join(latent_dir, class_folder)
            image_class_folder = os.path.join(image_dir, class_folder)
            for file in os.listdir(image_class_folder):
                self.dataset.append(
                    dict(
                        latent=os.path.join(latent_class_folder, file.split('.')[0]+'.safetensors'),
                        image=os.path.join(image_class_folder, file)
                    )
                )
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_item = dict()
        data = load_file(self.dataset[idx]['latent'])
        image = self.transform(Image.open(self.dataset[idx]['image']).convert("RGB"))
        if torch.rand(1) < self.RandomHorizontalFlipProb:
            data_item['latent'] = data['latent'][0]
            data_item['image'] = image[0]
        else:
            data_item['latent'] = data['latent'][1]
            data_item['image'] = image[1]
        data_item['label'] = data['label']
        return data_item



class C2ILoader():
    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config.dataloader.batch_size
        self.num_workers = data_config.dataloader.num_workers

        self.data_type = data_config.data_type
    
        if data_config.data_type == 'image':
            self.train_dataset = ImagenetDictWrapper(**OmegaConf.to_container(data_config.dataset))
        elif data_config.data_type == 'latent':
            self.train_dataset = ImagenetLatentDataset(**OmegaConf.to_container(data_config.dataset))
        else:
            raise NotImplementedError
        
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, rank, world_size, global_batch_size, max_steps, resume_steps, seed):
        
        sampler = get_train_sampler(
            self.train_dataset, rank, world_size, global_batch_size, max_steps, resume_steps, seed
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
        
    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )




