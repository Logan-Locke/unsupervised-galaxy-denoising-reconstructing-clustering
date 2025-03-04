import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data_loaders(base_path='/projects/dsci410_510/gz_candels_tensors', batch_size=256):
    """
    Create data loaders for train and test sets (data source came pre-split into train and test sets)

    Args:
        base_path (string): Base path to the dataset
        batch_size (int): Batch size for the dataloaders
    """

    # Load the image tensors
    train_images = torch.load(os.path.join(base_path, 'train_images_tensor.pt'))
    test_images = torch.load(os.path.join(base_path, 'test_images_tensor.pt'))

    # Create datasets where inputs and targets are the same
    train_dataset = TensorDataset(
        train_images,
        train_images
    )

    test_dataset = TensorDataset(
        test_images,
        test_images
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# Example usage:
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()

    # Print sample batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
