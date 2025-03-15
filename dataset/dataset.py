import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

import galsim
from galaxy_datasets.pytorch.datasets import GZHubble
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.version.cuda)

"""
# -------------------------------------------------
# Load in datasets and create a combined catalog
# -------------------------------------------------

root_dir = '/projects/dsci410_510/gz_hubble'

# Download original train/test subsets
train_dataset = GZHubble(root=root_dir, train=True, download=True, transform=None)
test_dataset = GZHubble(root=root_dir, train=False, download=True, transform=None)

# Extract train/test catalogs
train_catalog = train_dataset.catalog
test_catalog = test_dataset.catalog

print("Original train catalog shape:", train_catalog.shape)
print("Original test catalog shape:", test_catalog.shape)

# Combine the catalogs
combined_catalog = pd.concat([train_catalog, test_catalog], ignore_index=True)
combined_catalog_loc = os.path.join(root_dir, "full_catalog.parquet")
combined_catalog.to_parquet(combined_catalog_loc, index=False)

print("\nSaved the combined catalog to:", combined_catalog_loc)
"""

# -------------------------------
# Load in the combined catalog
# -------------------------------

root_dir = '/projects/dsci410_510/gz_hubble'
combined_catalog_loc = os.path.join(root_dir, "full_catalog.parquet")
full_catalog = pd.read_parquet(combined_catalog_loc)


# -----------------------------------------
# Masking function using Otsu's method
# ----------------------------------------

def create_background_mask_otsu(tensor_img):
    # Convert to grayscale
    r, g, b = tensor_img[0], tensor_img[1], tensor_img[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance coefficients

    # Move to CPU, convert to NumPy
    gray_np = gray.mul(255).to(torch.uint8).cpu().numpy()

    # Optimal threshold automatically determined based on the intensity histogram
    otsu_thresh_val, mask_uint8 = cv2.threshold(
        gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological operations to adjust mask
    kernel_open = np.ones((2, 2), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open, iterations=5)
    kernel_close = np.ones((4, 4), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    kernel_dilate = np.ones((4, 4), np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel_dilate, iterations=4)

    # Re-binarize
    cleaned_mask_np = (mask_uint8 > 0).astype(np.uint8)
    cleaned_mask_np = 1 - cleaned_mask_np

    # Convert back to float
    cleaned_mask = torch.from_numpy(cleaned_mask_np).float()
    return cleaned_mask


# -------------------------------
# Image de-normalizing function
# -------------------------------

denorm_mean = [0.0441, 0.0464, 0.0484]
denorm_std = [0.0712, 0.0726, 0.0713]
denorm_mean_tensor = torch.tensor(denorm_mean).view(1, -1, 1, 1)
denorm_std_tensor = torch.tensor(denorm_std).view(1, -1, 1, 1)


def denormalize_tensor(tensor, mean_tensor=denorm_mean_tensor, std_tensor=denorm_std_tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.device.type == 'cpu':
        mean_tensor = mean_tensor.cpu()
        std_tensor = std_tensor.cpu()
    else:
        mean_tensor = mean_tensor.to(tensor.device)
        std_tensor = std_tensor.to(tensor.device)

    denorm = tensor * std_tensor + mean_tensor

    return denorm.squeeze(0) if denorm.size(0) == 1 else denorm


# ------------------------------
# Create GalaxyDataModule
# --------------------------

# Workaround for odd API behavior related to initializing without transforms
def identity_transform(x):
    return x


# ------------------------------
# Transformation pipelines
# ------------------------------

# Custom noise transform using GalSim
class AddNoiseTransform:
    def __init__(self, noise_params):
        self.noise_params = noise_params
        if "rng" not in self.noise_params or self.noise_params["rng"] is None:
            self.noise_params["rng"] = galsim.BaseDeviate()

    def __call__(self, pil_img):
        # Convert PIL image to numpy array
        arr = np.asarray(pil_img, dtype=np.float32)
        H, W, C = arr.shape
        noisy_arr = np.empty_like(arr)

        # Process each channel individually (GalSim expects a 2D array)
        for ch in range(C):
            channel_arr = arr[:, :, ch]
            galsim_img = galsim.Image(channel_arr)

            poiss_noise = galsim.PoissonNoise(
                rng=self.noise_params["rng"],
                sky_level=self.noise_params.get("sky_level", 0.0),
            )
            gauss_noise = galsim.GaussianNoise(
                rng=self.noise_params["rng"],
                sigma=self.noise_params.get("sigma", 0.0)
            )

            # Poisson noise first, then Gaussian noise
            galsim_img.addNoise(poiss_noise)
            galsim_img.addNoise(gauss_noise)

            noisy_arr[:, :, ch] = np.clip(galsim_img.array, 0, 255)
        noisy_arr = noisy_arr.astype(np.uint8)

        return Image.fromarray(noisy_arr)


# Noise parameters
noise_params = {
    "rng"      : galsim.BaseDeviate(),
    "sky_level": 150.0,  # Poisson
    "sigma"    : 10.0,  # Gaussian
}

# Convert PIL → Tensor → float
base_transform = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
]
)

# Normalize transform
normalize_transform = T.Normalize(mean=denorm_mean, std=denorm_std)

# GalSim noise transform
noise_transform = AddNoiseTransform(noise_params)

# Geometric transform
random_geom_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
]
)


# Transform a pair of images
class SingleViewTransform:
    def __init__(self, random_geom_transform, base_transform, noise_transform, normalize_transform):
        self.random_geom_transform = random_geom_transform
        self.base_transform = base_transform
        self.noise_transform = noise_transform
        self.normalize_transform = normalize_transform

    def __call__(self, pil_img):
        # Apply random geometric transforms
        aug_img = self.random_geom_transform(pil_img)

        # Create "clean" image
        clean_tensor = self.base_transform(aug_img)
        clean_tensor = self.normalize_transform(clean_tensor)

        # Create "noisy" image
        noisy_pil = self.noise_transform(aug_img)
        noisy_tensor = self.base_transform(noisy_pil)
        noisy_tensor = self.normalize_transform(noisy_tensor)

        return clean_tensor, noisy_tensor


# Transform two pairs of images
class DoubleViewTransform:
    def __init__(self, single_view_transform):
        self.single_view_transform = single_view_transform

    def __call__(self, pil_img):
        clean1, noisy1 = self.single_view_transform(pil_img)
        clean2, noisy2 = self.single_view_transform(pil_img)

        return clean1, noisy1, clean2, noisy2


# ------------------------------
# Wrap the DataModule datasets
# ------------------------------

class DenoisingContrastiveDataset(Dataset):
    def __init__(
            self, base_dataset, double_view_transform
    ):
        self.base_dataset = base_dataset
        self.double_view_transform = double_view_transform
        self.to_pil = T.ToPILImage()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        raw = sample[0] if isinstance(sample, (tuple, list)) else sample

        # Error handling
        if not isinstance(raw, Image.Image):
            if isinstance(raw, torch.Tensor):
                pil_img = self.to_pil(raw)
            else:
                pil_img = Image.fromarray(raw.astype(np.uint8))
        else:
            pil_img = raw

        # Ensure RGB
        pil_img = pil_img.convert("RGB")

        # Produce the 4 images: (C1, N1, C2, N2)
        clean1, noisy1, clean2, noisy2 = self.double_view_transform(pil_img)

        # Create background masks
        denorm_clean1 = denormalize_tensor(clean1.clone())
        mask1 = create_background_mask_otsu(denorm_clean1)
        denorm_clean2 = denormalize_tensor(clean2.clone())
        mask2 = create_background_mask_otsu(denorm_clean2)

        return noisy1, noisy2, clean1, clean2, mask1, mask2


single_view_transform = SingleViewTransform(
    random_geom_transform=random_geom_transform,
    base_transform=base_transform,
    noise_transform=noise_transform,
    normalize_transform=normalize_transform
)
double_view_transform = DoubleViewTransform(single_view_transform)


# ---------------------------------------
# Create the datasets and dataloaders
# ---------------------------------------

def get_data_loaders(
        batch_size=64, train_fraction=0.7, val_fraction=0.1, test_fraction=0.2, num_workers=4,
        prefetch_factor=4, return_datasets=False
        ):
    datamodule = GalaxyDataModule(
        label_cols=None,
        catalog=full_catalog,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        custom_torchvision_transform=(identity_transform, identity_transform),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule)

    datamodule.train_dataset = DenoisingContrastiveDataset(
        base_dataset=datamodule.train_dataset,
        double_view_transform=double_view_transform,
    )
    datamodule.val_dataset = DenoisingContrastiveDataset(
        base_dataset=datamodule.val_dataset,
        double_view_transform=double_view_transform,
    )
    datamodule.test_dataset = DenoisingContrastiveDataset(
        base_dataset=datamodule.test_dataset,
        double_view_transform=double_view_transform,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if return_datasets:
        return datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset
    else:
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    batch_size = 64
    train_fraction = 0.7
    val_fraction = 0.1
    test_fraction = 0.2
    num_workers = 8
    prefetch_factor = 8

    train_loader, val_loader, test_loader = get_data_loaders(batch_size, train_fraction,
                                                             val_fraction, test_fraction,
                                                             num_workers, prefetch_factor
                                                             )

    # Send mean and std tensors to device after cpu is done with them
    denorm_mean_tensor = denorm_mean_tensor.to(device)
    denorm_std_tensor = denorm_std_tensor.to(device)

    noisy1_batch, _, _, _, _, _ = next(iter(train_loader))
    print("\nBatch shape:", noisy1_batch.shape)

    # Print sample batch
    """for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break"""
