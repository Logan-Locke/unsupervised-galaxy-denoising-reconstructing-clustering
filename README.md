# Galaxy Zoo CANDELS Project

This is my repository for the DSCI 410 Final Project.

## Project Overview

This project uses the Galaxy Zoo: Hubble (or GZ:H) dataset to de-noise, reconstruct, and cluster images of galaxies using their visual features via an autoencoder and contrastive learning.

## Data Overview

The dataset was created using data collected by the Advanced Camera for Surveys (ACS) aboard the Hubble Space Telescope (HST) and a community of more than 80,000
volunteers who classified galaxies. 
It contains nearly 120,000 images of galaxies, each with an associated identifier, metadata, and vote fractions generated from the volunteers.
Although the volunteer data was not used in this project, it may be useful to incorproate few-shot learning techniques or examine the clusters more closely.

<img alt="Data Sample" height="600" src="assets/data_sample.png" width="600"/>

## Methods Overview

The core component of this model is a autoencoder, similar to U-Net, with a projection head attatched that's used for contrastive learning.

The autoencoder using binary image masking to separate the foreground (the galaxy) from the background, which is then used to calculate separate loss functions for each component.
The background of each image uses a combination of total variation loss and a "darkening" loss. The foreground uses SSIM (Structural Similarity Index Measure) loss.
The total autoencoder loss is the combination of the background and foreground losses.

For the contrastive learning, NT-Xent (Normalized Temperature-Scaled Cross-Entropy) loss is used, which is then combined with the total autoencoder loss to obtain the loss for the entire model.

The process is as follows:
1. Take a "clean" image (C)
2. Duplicate C and apply random geometric transforms (C1, C2)
3. Duplicate C1 and C2 and add artificial noise (Poisson + Gaussian) to get the "noisy" images (N1, N2)
4. Using C1 and C2, calculate the binary image masks (M1, M2)

Autoencoder:
- Feed in N1 --> reconstruct C1
- Feed in N2 --> reconstruct C2

Contrastive Learning:
- Treat N1 and N2 as the positive pair
- Since it is unsupervised/self-supervised, we do not have any explicit negative pairs
  - The other images are implicitly treated as such

## Results

Denoising/Reconstruction:
- Foreground SSIM score of ~0.96
  - Value of 1 is considered "perfect"
 
Clustering:
- HDBSCAN with three clusters:
  - Silhouette Score of ~0.48
- K-means with eight clusters:
  - Silhouette Score of ~0.23
- Values range from -1 to +1, where:
  - 0.25 = "weak"
  - 0.50 = "moderate"
  - 0.70 = "strong"
  

## Conclusion

- Reconstructions were consistently really good
- Denoising was solid with some minor imperfections/artifacts
- Clustering/latent space representations were decent considering it did not have access to labeled data
