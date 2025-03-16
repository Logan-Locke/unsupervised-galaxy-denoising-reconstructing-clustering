print('Importing packages in train_models.py...')
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset.dataset import (
    device_checker,
    load_original_datasets,
    create_transforms,
    get_data_loaders,
)
from model.models import *
print('Done importing in train_models.py.')


# ------------------
# Helper functions
# ------------------

def get_unique_model_path(model_class_name, base_dir="model/saved_models", ext=".pt"):
    os.makedirs(base_dir, exist_ok=True)
    base_path = os.path.join(base_dir, model_class_name)
    file_path = base_path + ext
    counter = 1

    while os.path.exists(file_path):
        file_path = f"{base_path}_{counter}{ext}"
        counter += 1

    return file_path


def compute_batch_losses(
        model, noisy1, noisy2, clean1, clean2, mask1, mask2,
        lambda_galaxy, lambda_background, contrast_weight, lambda_temperature
        ):
    # Forward pass: obtain reconstructions and latent representations
    recon1, latent1 = model(noisy1, return_latent=True)
    recon2, latent2 = model(noisy2, return_latent=True)

    # Compute reconstruction, galaxy, and background losses for each augmented view
    rec_total1, gal_loss1, bg_loss1 = combined_autoencoder_loss(
        recon1, clean1, mask1, lambda_galaxy, lambda_background
    )
    rec_total2, gal_loss2, bg_loss2 = combined_autoencoder_loss(
        recon2, clean2, mask2, lambda_galaxy, lambda_background
    )

    # Average the losses from the two views
    loss_rec = (rec_total1 + rec_total2) / 2.0
    gal_loss = (gal_loss1 + gal_loss2) / 2.0
    bg_loss = (bg_loss1 + bg_loss2) / 2.0

    # Compute contrastive loss
    loss_contrast = nt_xent_loss(latent1, latent2, temperature=lambda_temperature)

    # Total loss (for training, this is the loss we backpropagate)
    total_loss = loss_rec + (contrast_weight * loss_contrast)

    return {
        "total"   : total_loss,
        "rec"     : loss_rec,
        "gal"     : gal_loss,
        "bg"      : bg_loss,
        "contrast": loss_contrast
    }


def train_model(
        model, train_loader, optimizer, lambda_galaxy, lambda_background,
        contrast_weight, lambda_temperature, early_stop=None
        ):
    model.train()
    running = {"total": 0.0, "rec": 0.0, "gal": 0.0, "bg": 0.0, "contrast": 0.0}
    num_batches = 0

    for batch_idx, (noisy1, noisy2, clean1, clean2, mask1, mask2) in enumerate(tqdm(train_loader, desc="Training...", leave=False)):
        if early_stop is not None and batch_idx >= early_stop:
            break

        num_batches += 1

        # Move batch to device
        noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
        clean1, clean2 = clean1.to(device), clean2.to(device)
        mask1, mask2 = mask1.to(device), mask2.to(device)

        optimizer.zero_grad()
        losses = compute_batch_losses(model, noisy1, noisy2, clean1, clean2, mask1, mask2,
                                      lambda_galaxy, lambda_background, contrast_weight,
                                      lambda_temperature
                                      )
        losses["total"].backward()
        optimizer.step()

        # Accumulate metrics
        for key in running:
            running[key] += losses[key].item()

    metrics = {k: (v / num_batches) for k, v in running.items()}
    return metrics


def evaluate_model(
        model, data_loader, lambda_galaxy, lambda_background,
        contrast_weight, lambda_temperature, early_stop=None
        ):
    model.eval()
    running = {"total": 0.0, "rec": 0.0, "gal": 0.0, "bg": 0.0, "contrast": 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (noisy1, noisy2, clean1, clean2, mask1, mask2) in enumerate(data_loader):
            if early_stop is not None and batch_idx >= early_stop:
                break

            num_batches += 1

            # Move batch to device
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            clean1, clean2 = clean1.to(device), clean2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)

            losses = compute_batch_losses(model, noisy1, noisy2, clean1, clean2, mask1, mask2,
                                          lambda_galaxy, lambda_background, contrast_weight,
                                          lambda_temperature
                                          )
            for key in running:
                running[key] += losses[key].item()

    metrics = {k: (v / num_batches) for k, v in running.items()}
    return metrics


# ------------------
# Training loop
# ------------------

def training_loop(
        model, num_epochs, learning_rate, lambda_galaxy, lambda_background, contrast_weight,
        contrast_temperature, train_loader, val_loader, early_stop=None, log_train_val_history=True,
        save_model=True
):
    model.apply(init_weights_xav).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metric histories
    train_history = {"total": [], "rec": [], "gal": [], "bg": [], "contrast": []}
    val_history = {"total": [], "rec": [], "gal": [], "bg": [], "contrast": []}

    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_model(
            model, train_loader, optimizer, lambda_galaxy, lambda_background, contrast_weight,
            contrast_temperature, early_stop
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train - Total: {train_metrics['total']:.3f} | Rec: {train_metrics['rec']:.3f} | "
            f"Gal: {train_metrics['gal']:.3f} | BG: {train_metrics['bg']:.3f} | Contrast: {train_metrics['contrast']:.3f}"
        )

        if log_train_val_history:
            print(f"Evaluating on validation set...")
            val_metrics = evaluate_model(
                model, val_loader, lambda_galaxy, lambda_background, contrast_weight,
                contrast_temperature, early_stop
            )

            # Store each metric
            for key in train_history:
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])

            print(
                f"Epoch {epoch+1}/{num_epochs} | Validation - Total: {val_metrics['total']:.3f} | Rec: {val_metrics['rec']:.3f} | "
                f"Gal: {val_metrics['gal']:.3f} | BG: {val_metrics['bg']:.3f} | Contrast: {val_metrics['contrast']:.3f}"
            )


    if save_model:
        save_path = get_unique_model_path(model.__class__.__name__)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    if log_train_val_history:
        return train_history, val_history


def plot_train_val_metrics(train_history, val_history):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_history['total']) + 1)

    plt.figure(figsize=(15, 5))

    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_history['total'], 'b-', label='Train')
    plt.plot(epochs, val_history['total'], 'r-', label='Validation')
    plt.title('Total Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.xticks(epochs)
    plt.legend()

    # Reconstruction Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_history['rec'], 'b-', label='Train')
    plt.plot(epochs, val_history['rec'], 'r-', label='Validation')
    plt.title('Reconstruction Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.xticks(epochs)
    plt.legend()

    # Contrast Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_history['contrast'], 'b-', label='Train')
    plt.plot(epochs, val_history['contrast'], 'r-', label='Validation')
    plt.title('Contrastive Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize
    device = device_checker()
    dataset_dir = 'data/gz_hubble'  # '/projects/dsci410_510/gz_hubble'
    full_catalog = load_original_datasets(dataset_dir)

    # Define transforms
    single_view_transform, double_view_transform = create_transforms(
        poisson=150.0,
        gaussian=10.0,
        random_crop=(0.5, 1.0),
        horizontal_flip=0.5,
        random_rot=45,
        color_jitter=(0.5, 0.5, 0.5, 0.1)
    )

    # Create loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        full_catalog,
        double_view_transform,
        batch_size=256,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        num_workers=8,
        prefetch_factor=8
    )

    unet_model = UNetContrastiveAutoencoder()
    custom_model = CustomContrastiveAutoencoder(
        activation_type='prelu',
        latent_dims=64
    )

    train_history, val_history = training_loop(
        model=unet_model,
        num_epochs=3,
        learning_rate=1e-3,
        lambda_galaxy=0.8,
        lambda_background=0.2,
        contrast_weight=0.75,
        contrast_temperature=0.75,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stop=None,
        log_train_val_history=True,
        save_model=True
    )

    # Plot the metrics
    plot_train_val_metrics(train_history, val_history)
