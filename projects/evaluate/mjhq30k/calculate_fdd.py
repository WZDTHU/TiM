
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import Normalize
from tqdm import tqdm




device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


class DINOv3Detector(torch.nn.Module):
    def __init__(self, use_patchtokens=True):
        super().__init__()
        encoder_dir = 'checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        encoder_name = encoder_dir.split('/')[-1].split('_pretrain')[0]
        self.encoder = torch.hub.load(
            'checkpoints/dinov3/dinov3', encoder_name, source='local', weights=encoder_dir,
        )
        self.encoder.to(device=device).eval().requires_grad_(False)
        self.norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).to(device)
        if use_patchtokens:
            def dino_forward(x):
                x = self.encoder.forward_features(self.norm(x))['x_norm_patchtokens']
                return torch.mean(x, dim=1)
        else:
            def dino_forward(x):
                return self.encoder(self.norm(x))
        self.compiled_encoder = torch.compile(dino_forward, mode="reduce-overhead")
        
    

    @torch.no_grad()
    def forward(self, x):
        return self.compiled_encoder(x)
            




class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            return None


def collate_fn(batch):
    """Filter out None values from batch"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.stack(batch)


def get_activations(files, model, height, width, dims=2048):
    batch_size = 50
    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.Compose([TF.ToTensor(), TF.Resize((height, width))]))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    pred_list = []  # Store predictions for valid images

    for batch in tqdm(dataloader):
        if batch is None:
            continue  # Skip empty batches (all images in batch were corrupted)
        
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        
        pred = pred.cpu().numpy()
        pred_list.append(pred)

    if len(pred_list) == 0:
        raise RuntimeError("No valid images found in the dataset")
    
    # Concatenate all predictions
    pred_arr = np.concatenate(pred_list, axis=0)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_statistics_of_path(path, model, height, width, dims):
    if path.endswith(".npz"):
        with np.load(path) as f:
            mu, sigma = f["mu"][:], f["sigma"][:]
    else:
        IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "JPEG"}
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.rglob("*.{}".format(ext))]
        )
        act = get_activations(files, model, height, width, dims)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
    return mu, sigma



def calculate_metrics_given_paths(paths, model, height, width, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    
    m1, s1 = compute_statistics_of_path(paths[0], model, height, width, dims)
    m2, s2 = compute_statistics_of_path(paths[1], model, height, width, dims)
    fdd_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fdd_value


def save_fdd_stats(paths, model, height, width, dims):
    """Saves FID statistics of one path"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    print(f"Saving statistics for {paths[0]}")

    mu, sigma = compute_statistics_of_path(paths[0], model, height, width, dims)
    
    np.savez_compressed(paths[1], mu=mu, sigma=sigma)





parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "paths",
    type=str,
    nargs=2,
    help=("Paths to the generated images or " "to .npz statistic files"),
)
parser.add_argument("--height", type=int, default=512, help="Batch size to use")
parser.add_argument("--width", type=int, default=512, help="Batch size to use")
parser.add_argument(
    "--save_stats",
    action="store_true",
    help=(
        "Generate an npz archive from a directory of "
        "samples. The first path is used as input and the "
        "second as output."
    ),
)
parser.add_argument(
    "--use_patchtokens",
    action="store_true",
    help=(
        "Generate an npz archive from a directory of "
        "samples. The first path is used as input and the "
        "second as output."
    ),
    default=False,
)

def main():
    args = parser.parse_args()
    model = DINOv3Detector(use_patchtokens=args.use_patchtokens)
    dims = model.encoder.embed_dim

    if args.save_stats:
        save_fdd_stats(args.paths, model, args.height, args.width, dims)
        return

    fdd_value = calculate_metrics_given_paths(
        args.paths, model, args.height, args.width, dims
    )
    print('reference: ', args.paths[0], 'evaluated: ', args.paths[1])
    print("FDD: ", fdd_value)



if __name__ == "__main__":
    main()