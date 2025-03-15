print('Importing packages in models.py...')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from dataset.dataset import denormalize_tensor, denorm_mean_tensor, denorm_std_tensor
print('Done importing in models.py.')


# -----------------------------
#  Weight initialization
# -----------------------------

def init_weights_xav(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -------------------------------
#  Autoencoder loss functions
# -------------------------------

ssim_loss_fn = SSIM(
    win_size=11,  # the size of gaussuain kernel
    win_sigma=1.5,  # sigma of normal distribution
    data_range=1.0,
    size_average=True,  # SSIM of all images will be averaged as a scalar
    channel=3
)


# Computes combined background loss to encourage smoothness and darkness
def combined_background_loss(image, bg_mask, darkening_weight):
    bg_mask = bg_mask.unsqueeze(1)

    # Total variation loss
    tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    tv_h_masked = tv_h * bg_mask[:, :, 1:, :]
    tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    tv_w_masked = tv_w * bg_mask[:, :, :, 1:]
    tv_loss = tv_h_masked.mean() + tv_w_masked.mean()

    # "Darkening" loss
    target_black = torch.zeros_like(image)
    darkening_loss = torch.nn.functional.mse_loss(image * bg_mask, target_black * bg_mask)

    # Combined background loss
    combined_background_loss_val = tv_loss + (darkening_weight * darkening_loss)
    return combined_background_loss_val


# Computes foreground/galaxy loss to encourage reconstruction detail
def galaxy_loss(reconstructed, target, galaxy_mask):
    galaxy_mask = galaxy_mask.unsqueeze(1)

    # Mask out background in both images so only foreground contributes to loss
    rec_galaxy = reconstructed * galaxy_mask
    tgt_galaxy = target * galaxy_mask

    # Compute SSIM-based loss for masked galaxy region
    galaxy_loss_val = 1.0 - ssim_loss_fn(rec_galaxy, tgt_galaxy)

    return galaxy_loss_val


# Computes the total loss for the autoencoder portion using the background and foreground/galaxy
# losses
def combined_autoencoder_loss(
        reconstructed, target, mask, lambda_galaxy=0.7, lambda_background=0.3, darkening_weight=0.25
):
    # Error check
    if lambda_galaxy + lambda_background != 1.0:
        raise ValueError("lambda_galaxy + lambda_background must equal 1.0")

    # Denormalize
    rec_denorm = denormalize_tensor(reconstructed, denorm_mean_tensor, denorm_std_tensor)
    tgt_denorm = denormalize_tensor(target, denorm_mean_tensor, denorm_std_tensor)

    # Compute background loss
    background_loss_val = combined_background_loss(rec_denorm, mask,
                                                   darkening_weight=darkening_weight
                                                   )

    # Compute foreground/galaxy loss
    galaxy_mask = 1 - mask
    galaxy_loss_val = galaxy_loss(rec_denorm, tgt_denorm, galaxy_mask)

    # Compute combined autoencoder loss
    combined_autoencoder_loss_val = (lambda_galaxy * galaxy_loss_val) + (
            lambda_background * background_loss_val)

    return combined_autoencoder_loss_val, galaxy_loss_val, background_loss_val


# --------------------------------------
#  Contrastive learning loss function
# --------------------------------------

# Computes NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss
def nt_xent_loss(z1, z2, temperature=0.5):
    # Get batch size
    batch_size = z1.shape[0]

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate embeddings
    z = torch.cat([z1, z2], dim=0)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(z, z.T)

    # Scale similarities by temperature
    logits = similarity_matrix / temperature

    # Mask out self-similarity
    diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    logits = logits.masked_fill(diagonal_mask, -1e9)

    # Construct labels
    labels = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (2 * batch_size)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


# ----------------------
#  Test set evaluation
# ----------------------

# Evaluates test metrics
def compute_test_metrics(model, test_loader, device, early_stop=None):
    # Initialize SSIM metric
    ssim_metric_denorm = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_metric_denorm.reset()

    model.eval()
    with torch.inference_mode():
        for batch_idx, (noisy1, noisy2, clean1, clean2, mask1, mask2) in enumerate(test_loader):
            if early_stop is not None and batch_idx >= early_stop:
                break

            # Move data to device
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            clean1, clean2 = clean1.to(device), clean2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)

            # Forward pass for both augmented views
            outputs1, _ = model(noisy1, return_latent=True)
            outputs2, _ = model(noisy2, return_latent=True)

            # Denormalize output and target images
            outputs_denorm1 = denormalize_tensor(outputs1, denorm_mean_tensor, denorm_std_tensor)
            targets_denorm1 = denormalize_tensor(clean1, denorm_mean_tensor, denorm_std_tensor)
            outputs_denorm2 = denormalize_tensor(outputs2, denorm_mean_tensor, denorm_std_tensor)
            targets_denorm2 = denormalize_tensor(clean2, denorm_mean_tensor, denorm_std_tensor)

            # Generate foreground/galaxy masks
            galaxy_mask1 = (1 - mask1).unsqueeze(1)
            galaxy_mask2 = (1 - mask2).unsqueeze(1)

            # Isolate the galaxy region
            outputs_galaxy1 = outputs_denorm1 * galaxy_mask1
            targets_galaxy1 = targets_denorm1 * galaxy_mask1
            outputs_galaxy2 = outputs_denorm2 * galaxy_mask2
            targets_galaxy2 = targets_denorm2 * galaxy_mask2

            # Update SSIM metric for both views
            ssim_metric_denorm.update(outputs_galaxy1, targets_galaxy1)
            ssim_metric_denorm.update(outputs_galaxy2, targets_galaxy2)

    # Compute overall SSIM score across both views
    test_ssim_score = ssim_metric_denorm.compute().item()

    return test_ssim_score


# -------------------------------------------
#  Function to switch activation functions
# -------------------------------------------

activations = {
    'relu'      : lambda: nn.ReLU(inplace=True),
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
    'prelu'     : lambda: nn.PReLU(num_parameters=1, init=0.25),
    'elu'       : lambda: nn.ELU(alpha=1.0, inplace=True),
    'selu'      : lambda: nn.SELU(inplace=True),
    'gelu'      : lambda: nn.GELU(),
    'softplus'  : lambda: nn.Softplus(beta=1, threshold=20),
}


def get_activation_fn(activation_type):
    try:
        return activations[activation_type]
    except KeyError:
        raise ValueError("Unknown activation type")

# -------------------------------
# Custom autoencoder components
# -------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn_factory, kernel_size=3, stride=2,
            padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            activation_fn_factory()
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(
            self, in_channels, skip_channels, out_channels, activation_fn_factory,
            kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation_fn_factory()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ----------------------------
# Custom autoencoder model
# ----------------------------

class CustomAutoencoder(nn.Module):
    def __init__(self, activation_type, latent_dim=128, activation_fn_factory=None):
        super().__init__()

        if activation_fn_factory is None:
            activation_fn_factory = get_activation_fn(activation_type)
        self.latent_dim = latent_dim

        # Encoder blocks
        self.eb1 = EncoderBlock(3, 16, activation_fn_factory)
        self.eb2 = EncoderBlock(16, 32, activation_fn_factory)
        self.eb3 = EncoderBlock(32, 64, activation_fn_factory)
        self.eb4 = EncoderBlock(64, 128, activation_fn_factory)

        # Pooling for latent vector computation
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Latent mapping
        self.latent_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, latent_dim),
            activation_fn_factory()
        )

        # Latent expansion
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            activation_fn_factory()
        )

        # Decoder blocks w/ skip connections
        self.db1 = DecoderBlock(128, skip_channels=128, out_channels=128, activation_fn_factory=activation_fn_factory)
        self.db2 = DecoderBlock(128, skip_channels=64, out_channels=64, activation_fn_factory=activation_fn_factory)
        self.db3 = DecoderBlock(64, skip_channels=32, out_channels=32, activation_fn_factory=activation_fn_factory)
        self.db4 = DecoderBlock(32, skip_channels=16, out_channels=16, activation_fn_factory=activation_fn_factory)

        #  Final upsampling
        self.final_upsample = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

        # Projection head for contrastive loss
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
        )

    def encode(self, x):
        feat1 = self.eb1(x)                   # (batch, 16, 112, 112)
        feat2 = self.eb2(feat1)               # (batch, 32, 56, 56)
        feat3 = self.eb3(feat2)               # (batch, 64, 28, 28)
        feat4 = self.eb4(feat3)               # (batch, 128, 14, 14)

        pooled = self.avg_pool(feat4)         # (batch, 128, 7, 7)
        latent = self.latent_encoder(pooled)

        return latent, pooled, (feat1, feat2, feat3, feat4)

    def decode(self, latent, encoder_features):
        x = self.latent_decoder(latent)
        x = x.view(-1, 128, 7, 7)

        x = self.db1(x, encoder_features[3])    # (batch, 128, 14, 14)
        x = self.db2(x, encoder_features[2])    # (batch, 64, 28, 28)
        x = self.db3(x, encoder_features[1])    # (batch, 32, 56, 56)
        x = self.db4(x, encoder_features[0])    # (batch, 16, 112, 112)

        reconstructed = self.final_upsample(x)  # (batch, 3, 224, 224)

        return reconstructed

    def forward(self, x, return_latent=False):
        latent, pooled, encoder_features = self.encode(x)
        reconstruction = self.decode(latent, encoder_features)
        proj = self.projection_head(pooled)

        if return_latent:
            return reconstruction, proj
        return reconstruction


# ------------------------------------
# U-Net-style autoencoder components
# ------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # After concatenation, the number of channels becomes in_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Concatenate skip connection from encoder with upsampled features
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class FinalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------------
# U-Net-style autoencoder model
# -------------------------------

class UNetAutoencoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, latent_dim=128):
        super(UNetAutoencoder, self).__init__()

        # Initial double convolution
        self.initial_conv = DoubleConv(n_channels, 64)

        # Downsampling blocks
        self.down1 = DownsamplingBlock(64, 128)
        self.down2 = DownsamplingBlock(128, 256)
        self.down3 = DownsamplingBlock(256, 256)

        # Upsampling blocks
        self.up1 = UpsamplingBlock(256 + 256, 128)
        self.up2 = UpsamplingBlock(128 + 128, 64)
        self.up3 = UpsamplingBlock(64 + 64, 64)

        # Final output convolution
        self.final_conv = FinalConvLayer(64, n_classes)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, return_latent=True):
        # Encoder
        x1 = self.initial_conv(x)  # (batch, 64, 224, 224)
        x2 = self.down1(x1)        # (batch, 128, 112, 112)
        x3 = self.down2(x2)        # (batch, 256, 56, 56)
        x4 = self.down3(x3)        # (batch, 256, 28, 28)

        # Decoder
        x = self.up1(x4, x3)       # (batch, 128, 56, 56)
        x = self.up2(x, x2)        # (batch, 64, 112, 112)
        x = self.up3(x, x1)        # (batch, 64, 224, 224)
        output = self.final_conv(x)

        # Projection head from bottleneck feature
        latent_proj = self.projection_head(x4)

        if return_latent:
            return output, latent_proj
        return output
