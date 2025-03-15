print('Importing packages in train_models.py...')
import torch.optim as optim
from tqdm import tqdm
from dataset.dataset import (
    device_checker,
    load_original_datasets,
    create_transforms,
    get_data_loaders,
)
from models import *
print('Done importing in train_models.py.')


# ------------------
# Training loop
# ------------------

def train_model(
        model, num_epochs, learning_rate, lambda_galaxy, lambda_background, lambda_contrast,
        lambda_temperature, train_loader, early_stop=None, save_model=True
):
    model.apply(init_weights_xav).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        # Running sums to track losses across batches
        running_total_loss = 0.0
        running_rec_loss = 0.0
        running_gal_loss = 0.0
        running_bg_loss = 0.0
        running_con_loss = 0.0

        num_batches = 0

        for batch_idx, (noisy1, noisy2, clean1, clean2, mask1, mask2) in enumerate(tqdm(train_loader)):
            if early_stop is not None:
                if batch_idx >= early_stop:
                    break

            num_batches += 1

            # Move data to device
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            clean1, clean2 = clean1.to(device), clean2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)

            # Forward pass
            recon1, latent1 = model(noisy1, return_latent=True)
            recon2, latent2 = model(noisy2, return_latent=True)

            # Compute reconstruction loss for each augmented view
            rec_total1, gal_loss1, bg_loss1 = combined_autoencoder_loss(
                recon1,
                clean1,
                mask1,
                lambda_galaxy,
                lambda_background
            )
            rec_total2, gal_loss2, bg_loss2 = combined_autoencoder_loss(
                recon2,
                clean2,
                mask2,
                lambda_galaxy,
                lambda_background
            )

            # Average reconstruction losses across both views
            loss_rec = (rec_total1 + rec_total2) / 2.0
            gal_loss = (gal_loss1 + gal_loss2) / 2.0
            bg_loss = (bg_loss1 + bg_loss2) / 2.0

            # Contrastive loss
            loss_contrast = nt_xent_loss(latent1, latent2, temperature=lambda_temperature)

            # Total loss
            loss = loss_rec + (lambda_contrast * loss_contrast)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch losses
            running_total_loss += loss.item()
            running_rec_loss += loss_rec.item()
            running_gal_loss += gal_loss.item()
            running_bg_loss += bg_loss.item()
            running_con_loss += loss_contrast.item()

        # Compute epoch averages
        avg_total_loss = running_total_loss / num_batches
        avg_rec_loss = running_rec_loss / num_batches
        avg_gal_loss = running_gal_loss / num_batches
        avg_bg_loss = running_bg_loss / num_batches
        avg_con_loss = running_con_loss / num_batches

        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Total: {avg_total_loss:.3f} | Rec: {avg_rec_loss:.3f} | "
            f"Gal: {avg_gal_loss:.3f} | BG: {avg_bg_loss:.3f} | Contrast: {avg_con_loss:.3f}"
        )

    if save_model:
        torch.save(model.state_dict(), f"model/saved_models/{model.__class__.__name__}.pt")
        print(f"Model saved to model/saved_models/{model.__class__.__name__}.pt")

# avg_test_combined_autoencoder_loss, avg_test_background_loss, avg_test_galaxy_loss,
# test_galaxy_ssim_score = evaluate_test_metrics()
# print(f"\n(Final Test) Combined Loss: {avg_test_combined_autoencoder_loss:.3f} | Galaxy SSIM
# Score: {test_galaxy_ssim_score:.3f}")

if __name__ == "__main__":
    # Initialize
    device = device_checker()
    dataset_dir = 'data/gz_hubble'  # UPDATE THIS
    full_catalog = load_original_datasets(dataset_dir)

    # **IF USING GPU, UPDATE DATALOADER PARAMETERS**

    # Create dataloaders
    single_view_transform, double_view_transform = create_transforms()
    train_loader, val_loader, test_loader = get_data_loaders(
        full_catalog,
        double_view_transform,
        batch_size=4,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        num_workers=0,
        prefetch_factor=0
    )

    unet_model = UNetAutoencoder()
    custom_model = CustomAutoencoder(
        activation_type='prelu',
        latent_dim=32
    )

    train_model(
        model=custom_model,
        num_epochs=3,
        learning_rate=1e-3,
        lambda_galaxy=0.8,
        lambda_background=0.2,
        lambda_contrast=0.75,
        lambda_temperature=0.75,
        train_loader=train_loader,
        early_stop=1,
        save_model=True
    )
