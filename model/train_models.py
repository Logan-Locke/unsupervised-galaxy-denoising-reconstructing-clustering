import torch
import torch.optim as optim
from tqdm import tqdm

from dataset.dataset import train_loader, val_loader, test_loader
from models import *


# ------------------
# Training loop
# ------------------

def train_model(
        NUM_EPOCHS, ACTIVATION_TYPE, LATENT_DIM, LEARNING_RATE, LAMBDA_GALAXY,
        LAMBDA_BACKGROUND, LAMBDA_CONTRAST, TEMPERATURE, COMPUTE_TEST_METRICS, model,
        optimizer, train_loader, val_loader, test_loader, device=device, save_model=True
):
    for epoch in range(NUM_EPOCHS):
        model.train()

        # Running sums to track losses across batches
        running_total_loss = 0.0
        running_rec_loss = 0.0
        running_gal_loss = 0.0
        running_bg_loss = 0.0
        running_con_loss = 0.0

        num_batches = 0

        for batch_idx, (noisy1, noisy2, clean1, clean2, mask1, mask2) in enumerate(tqdm(train_loader)):
            # Optional early stopping
            if batch_idx >= 100:
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
                LAMBDA_GALAXY,
                LAMBDA_BACKGROUND
            )
            rec_total2, gal_loss2, bg_loss2 = combined_autoencoder_loss(
                recon2,
                clean2,
                mask2,
                LAMBDA_GALAXY,
                LAMBDA_BACKGROUND
            )

            # Average reconstruction losses across both views
            loss_rec = (rec_total1 + rec_total2) / 2.0
            gal_loss = (gal_loss1 + gal_loss2) / 2.0
            bg_loss = (bg_loss1 + bg_loss2) / 2.0

            # Contrastive loss
            loss_contrast = nt_xent_loss(latent1, latent2, temperature=TEMPERATURE)

            # Total loss
            loss = loss_rec + (LAMBDA_CONTRAST * loss_contrast)

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
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Total: {avg_total_loss:.3f} | Rec: {avg_rec_loss:.3f} | "
            f"Gal: {avg_gal_loss:.3f} | BG: {avg_bg_loss:.3f} | Contrast: {avg_con_loss:.3f}"
        )

# avg_test_combined_autoencoder_loss, avg_test_background_loss, avg_test_galaxy_loss,
# test_galaxy_ssim_score = evaluate_test_metrics()
# print(f"\n(Final Test) Combined Loss: {avg_test_combined_autoencoder_loss:.3f} | Galaxy SSIM
# Score: {test_galaxy_ssim_score:.3f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.version.cuda)

    NUM_EPOCHS = 3
    ACTIVATION_TYPE = 'prelu'
    LATENT_DIM = 256
    LEARNING_RATE = 1e-3
    LAMBDA_GALAXY = 0.8
    LAMBDA_BACKGROUND = 0.2
    LAMBDA_CONTRAST = 0.75
    TEMPERATURE = 0.75
    COMPUTE_TEST_METRICS = False

    # Initialize model, weights, and optimizer
    model = UNetAutoencoder().apply(init_weights_xav).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(NUM_EPOCHS, ACTIVATION_TYPE, LATENT_DIM, LEARNING_RATE, LAMBDA_GALAXY,
                LAMBDA_BACKGROUND, LAMBDA_CONTRAST, TEMPERATURE, COMPUTE_TEST_METRICS, model,
                optimizer, train_loader, val_loader, test_loader, device=device, save_model=True)
