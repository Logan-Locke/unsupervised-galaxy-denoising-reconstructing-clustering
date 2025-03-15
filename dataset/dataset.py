print('Importing...')
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
print('Done importing.\n')

def device_checker():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}\n')

    return device


# -------------------------------------------------
# Load in datasets and create a combined catalog
# -------------------------------------------------

def load_original_datasets(dataset_dir, fresh_download=False, return_og_catalogs=False):
    if fresh_download:
        print('Downloading original datasets...')
        og_train_dataset = GZHubble(root=dataset_dir, train=True, download=True, transform=None)
        og_test_dataset = GZHubble(root=dataset_dir, train=False, download=True, transform=None)
    else:
        print('Loading original datasets from file...')
        og_train_dataset = GZHubble(root=dataset_dir, train=True, download=False, transform=None)
        og_test_dataset = GZHubble(root=dataset_dir, train=False, download=False, transform=None)

    # Extract train/tesst catalogs
    og_train_catalog = og_train_dataset.catalog
    og_test_catalog = og_test_dataset.catalog

    # Combine pre-split catalogs
    combined_catalog = pd.concat([og_train_catalog, og_test_catalog], ignore_index=True)
    combined_catalog_loc = os.path.join(dataset_dir, 'full_catalog.parquet')
    combined_catalog.to_parquet(combined_catalog_loc, index=False)
    print('Saved the new combined catalog to:', combined_catalog_loc)

    full_catalog = pd.read_parquet(combined_catalog_loc)

    if return_og_catalogs:
        return og_train_catalog, og_test_catalog, full_catalog
    else:
        return full_catalog


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
# Transformation pipelines
# ------------------------------

# Workaround for odd API behavior related to initializing without transforms
def identity_transform(x):
    return x

# Custom noise transform using GalSim
class AddNoiseTransform:
    def __init__(self, noise_params):
        self.noise_params = noise_params
        if 'rng' not in self.noise_params or self.noise_params['rng'] is None:
            self.noise_params['rng'] = galsim.BaseDeviate()

    def __call__(self, pil_img):
        arr = np.asarray(pil_img, dtype=np.float32)
        H, W, C = arr.shape
        noisy_arr = np.empty_like(arr)

        # Process each channel individually (GalSim expects a 2-D array)
        for ch in range(C):
            channel_arr = arr[:, :, ch]
            galsim_img = galsim.Image(channel_arr)

            poiss_noise = galsim.PoissonNoise(
                rng=self.noise_params['rng'],
                sky_level=self.noise_params.get('sky_level', 0.0),
            )
            gauss_noise = galsim.GaussianNoise(
                rng=self.noise_params['rng'],
                sigma=self.noise_params.get('sigma', 0.0)
            )

            # Poisson noise first, then Gaussian noise
            galsim_img.addNoise(poiss_noise)
            galsim_img.addNoise(gauss_noise)

            noisy_arr[:, :, ch] = np.clip(galsim_img.array, 0, 255)

        noisy_arr = noisy_arr.astype(np.uint8)

        return Image.fromarray(noisy_arr)


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
        pil_img = pil_img.convert('RGB')

        # Produce the 4 images: (C1, N1, C2, N2)
        clean1, noisy1, clean2, noisy2 = self.double_view_transform(pil_img)

        # Create background masks
        denorm_clean1 = denormalize_tensor(clean1.clone())
        mask1 = create_background_mask_otsu(denorm_clean1)
        denorm_clean2 = denormalize_tensor(clean2.clone())
        mask2 = create_background_mask_otsu(denorm_clean2)

        return noisy1, noisy2, clean1, clean2, mask1, mask2


# -------------------------------
# Combined transforms function
# -------------------------------

def create_transforms(poisson=100.0, gaussian=5.0, random_crop=(0.5, 1.0), horizontal_flip=0.5,
                      random_rot=45, color_jitter=(0.5, 0.5, 0.5, 0.1)):
    # Define noise parameters
    noise_params = {
        'rng'      : galsim.BaseDeviate(),
        'sky_level': poisson,
        'sigma'    : gaussian
    }

    # Initialize noise transform
    noise_transform = AddNoiseTransform(noise_params)

    # Convert PIL → Tensor → float
    base_transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
    ])

    # Normalize transform
    normalize_transform = T.Normalize(mean=denorm_mean, std=denorm_std)

    # Geometric transform
    random_geom_transform = T.Compose([
        T.RandomResizedCrop(224, scale=random_crop),
        T.RandomHorizontalFlip(p=horizontal_flip),
        T.RandomRotation(degrees=random_rot),
        T.ColorJitter(brightness=color_jitter[0], contrast=color_jitter[1],
                      saturation=color_jitter[2], hue=color_jitter[3])
    ])

    # Create single and double view transforms
    single_view_transform = SingleViewTransform(
        random_geom_transform=random_geom_transform,
        base_transform=base_transform,
        noise_transform=noise_transform,
        normalize_transform=normalize_transform
    )
    double_view_transform = DoubleViewTransform(single_view_transform)

    return single_view_transform, double_view_transform


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
    device = device_checker()

    # dataset_dir = '/projects/dsci410_510/gz_hubble'
    dataset_dir = 'data/gz_hubble' # UPDATE THIS

    full_catalog = load_original_datasets(
        dataset_dir,
        fresh_download=False,
        return_og_catalogs=False
    )

    # Create transforms
    print('Creating transforms...')
    single_view_transform, double_view_transform = create_transforms(
        poisson=150.0,
        gaussian=10.0,
        random_crop=(0.5, 1.0),
        horizontal_flip=0.5,
        random_rot=45,
        color_jitter=(0.5, 0.5, 0.5, 0.1)
    )
    print('Transforms created.\n')

    # IF ON GPU, UPDATE DATALOADER PARAMETERS
    # Create dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=64,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        num_workers=0,
        prefetch_factor=0
    )
    print('Dataloaders created.\n')

    # Send mean and std tensors to device
    denorm_mean_tensor = denorm_mean_tensor.to(device)
    denorm_std_tensor = denorm_std_tensor.to(device)

    noisy1, _, _, _, _, _ = next(iter(train_loader))
    print('\nBatch shape:', noisy1.shape)
