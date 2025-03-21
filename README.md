# Galaxy Zoo: Hubble Project

## Project Overview

This project uses the Galaxy Zoo: Hubble (or GZ:H) dataset to de-noise, reconstruct, and cluster
images of galaxies using their visual features via an autoencoder and contrastive learning.

## Data Overview

The dataset was created using data collected by the Advanced Camera for Surveys (ACS) aboard the
Hubble Space Telescope (HST) and a community of more than 80,000
volunteers who classified galaxies.
It contains nearly 120,000 images of galaxies, each with an
associated identifier, metadata, and vote fractions generated from the volunteers.
Although the
volunteer data was not used in this project, it may be useful to incorporate few-shot learning
techniques or examine the clusters more closely.

The original data from the Galaxy Zoo project can be
found [here](https://data.galaxyzoo.org/#section-11), and the GitHub repository I used to download
the images can be found [here](https://github.com/mwalmsley/galaxy-datasets).

<img alt="Data Sample" height="600" src="assets/data_sample.png" width="600"/>

## Model Architectures Overview

This repository includes two contrastive autoencoder models:
`CustomContrastiveAutoencoder` and `UNetContrastiveAutoencoder`.
Both models combine an autoencoding architecture with a projection head for contrastive learning,
but they differ in their design, reconstruction strategy, and customizability.

### Custom Contrastive Autoencoder:

- **Encoder:**
    - **EncoderBlock:** Four sequential blocks (Conv2d → BatchNorm2d → Activation) progressively reduce the spatial dimensions.
    - **Latent Mapping:** An AvgPool2d layer pools the final encoder output, which is then flattened and passed through a Linear layer (with activation) to create a latent vector.

- **Decoder:**
    - **Latent Expansion:** A Linear layer expands the latent vector back to a spatial map.
    - **DecoderBlock:** Four blocks that apply a ConvTranspose2d for upsampling, concatenate corresponding skip connections from the encoder, and refine the features using a Conv2d, BatchNorm2d, and Activation.
    - **Final Upsampling:** A ConvTranspose2d layer reconstructs the output image.

- **Contrastive Learning:**
    - **Projection Head:** Consists of AdaptiveAvgPool2d, Flatten, and Linear layers (with activation) to extract a latent representation for contrastive loss.

### U-Net-Style Contrastive Autoencoder

- **Encoder:**
   - **DoubleConv:** Begins with a DoubleConv block (two sequential Conv2d layers with BatchNorm2d and ReLU) to extract initial features.
   - **DownsamplingBlock:** Three blocks combine MaxPool2d with DoubleConv to reduce spatial dimensions while increasing feature depth.

- **Decoder:**
   - **UpsamplingBlock:** Three blocks perform bilinear upsampling, concatenate skip connections from the encoder, and apply DoubleConv for feature refinement.
   - **FinalConvLayer:** A 1×1 Conv2d produces the final reconstructed output.

- **Contrastive Learning:**
   - **Projection Head:** Attached to the bottleneck, it uses AdaptiveAvgPool2d, Flatten, and Linear layers (with ReLU) to generate the latent representation for contrastive tasks.

### Key Differences

The `CustomContrastiveAutoencoder` uses a conventional encoder composed of sequential convolutional blocks that progressively reduce the spatial dimensions.
This is followed by average pooling and a linear layer that flattens the features into a latent vector.
The latent vector is then expanded through another linear layer and decoded using transposed convolutions combined with skip connections that bring in corresponding encoder features.
In contrast, the `UNetContrastiveAutoencoder` employs a fully convolutional, UNet-style architecture
that preserves spatial details throughout the network by using DoubleConv blocks and max pooling for downsampling.
It derives its latent representation directly from the bottleneck features using a projection head, and its decoder reconstructs the image through bilinear upsampling with concatenated skip connections and additional convolutional operations.

### Loss Functions and Masking

These models leverage binary image masking to separate the foreground (the galaxy) from the background. This enables separate loss computations:

* **Foreground (Galaxy) Loss:** The `galaxy_loss` function uses SSIM loss to ensure perceptually detailed reconstruction of the galaxy itself.

* **Background Loss:** The `combined_background_loss` function applies total variation loss for smoothness and a “darkening” loss to enforce a darker background.

* **Total Autoencoder Loss:** The `combined_autoencoder_loss` function merges the foreground and background losses using configurable weights.

* **Contrastive Loss:** The `nt_xent_loss` function implements NT-Xent (Normalized Temperature-Scaled Cross-Entropy) loss on the projection head outputs, reinforcing robust latent representations.


### The Prerocessing Pipeline

1. Take a "clean" image (C)
2. Duplicate C and apply random geometric transforms (C1, C2)
3. Duplicate C1 and C2 and add artificial noise (Poisson + Gaussian) to get the "noisy" images (N1,
   N2)
4. Using C1 and C2, calculate the binary image masks (M1, M2)

Below is a look of what one component of the dataset looks like (two clean images, two noisy
images, and two masks).
For each of the images in the training dataset (~80,000), the following is created:

<img alt="Masking" height="400" src="assets/masking.png" width="700"/>

**A basic overview of the model steps:**

- **Autoencoder:**
    - Feed in N1 --> reconstruct C1
    - Feed in N2 --> reconstruct C2
- **Contrastive Learning:**
    - Treat N1 and N2 as the positive pair
    - Since it is unsupervised/self-supervised, we do not have any explicit negative pairs
        - The other images are implicitly treated as such

## Training the Model(s)

1. **Prepare the Environment:**
    1. Ensure all required packages are installed.
    2. The device should automatically be configured using `device_checker()`.
    3. Verify the dataset is either placed in the correct directory or choose to install it using `fresh_download=True` in `load_original_datasets()`.

2. **Configure Data:**
    1. The script automatically loads and preprocesses the data using `dataset.py`.
    2. Tune the noise and augmentations added to the images using the `create_transforms()` parameters.
    3. Tune the dataloaders using the `get_data_loaders()` parameters.
        1. **Note:** The U-Net-Style model is computationally demanding, start with smaller batch sizes and increment from there (if necessary).

3. **Configure Model:**
    1. The script automatically loads the models using `models.py`.
    2. Two model types are defined (`UNetContrastiveAAutoencoder` and `CustomContrastiveAAutoencoder`). Choose which one to train by selecting the appropriate model instance.
    3. For the custom model, you may set the output latent dimensions using `latent_dims`, and a specific activation function to be used in each layer using `activation_type`.
        1. The choices are the following: 'relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'gelu', and 'softplus'.

4. **Configure Training Hyperparameters:**
    1. General hyperparameters: `num_epochs`, `learning_rate`, and `early_stop`.
    2. Autoencoder hyperparameters that influence the loss weights: `lambda_galaxy` and `lambda_background`.
    3. Contrastive learning hyperparameters: `contrast_weight` and `contrast_temperature`.
        1. `contrast_weight` controls how much weight should be given to the contrastive loss in comparison to the reconstruction loss.
    4. Miscellaneous hyperparameters: `log_train_val_history` and `save_model`.
        1. **Note**: if `log_train_val_history=False`, you must remove the `train_history` and `val_history` assignments before calling `training_loop()`.

5. **Run the Training Loop:**
    1. Simply execute the script (i.e. run `train_models.py`).
    2. The training loop will iterate through the dataset, display progress, and (optionally) log performance metrics for each epoch using both the training and validation sets.
    3. After training, the model can be plotted to visualize trends over the training epochs using `plot_train_val_metrics()`
        1. **Note:** `log_train_val_history=True` must be set to plot.

## Results

### Denoising and Reconstruction:

- Galaxy SSIM score of ~0.995
    - Value of 1 is considered "perfect"

Fundamentally, autoencoders cannot reconstruct images that are cleaner than the target images—they
can only reduce the amount of noise to that which is present in the target images.
Thus, since we do
not have access to a subset of ultra-clean galaxy images, and we instead use *all* galaxy images for
the autoencoding, the autoencoder would normally only learn how to remove the artificial noise
introduced to the input images during the pre-processing stage.
That being said, through the clever
use of the binary image masks and specific loss functions, this model's autoencoder *does* learn how
to remove the noise found in the background of the image.
The noise that overlaps the galaxies
themselves still largely remains intact for the reasons mentioned above, however.

Below are some examples of how the model de-noises and reconstructs the input images:

<img alt="Denoising, Ex. 1" height="200" src="assets/denoising-1.png" width="550"/>
<img alt="Denoising, Ex. 3" height="200" src="assets/denoising-2.png" width="550"/>
<img alt="Denoising, Ex. 3" height="200" src="assets/denoising-3.png" width="550"/>

### Clustering:

- HDBSCAN with three clusters:
   - Calinski-Harabasz Index: ~5,800
   - Davies-Bouldin Index: ~1.15
   - Silhouette: ~0.41

- K-means with three clusters:
   - Calinski-Harabasz Index: ~4,900
   - Davies-Bouldin Index: ~1.11
   - Silhouette: ~0.66

Although K-means shows a higher Silhouette (~0.66) compared to HDBSCAN (~0.41), this metric can overestimate performance when every point is forced into a cluster. 
I found that by using K-means, one cluster overwhelmingly dominates, which suggests that the high silhouette score might not reflect a truly natural grouping.
Looking at the Calinski-Harabasz Index (CHI), HDBSCAN’s higher value (~5,800 vs. ~4,900) indicates that its clusters are better separated relative to their internal dispersion. 

While K-means appears better based on the Silhouette and marginally better DBI, these metrics can be misleading due to the imbalanced cluster sizes and forced grouping often seen in K-means. 
Because HDBSCAN naturally identifies outliers and produces more balanced clusters, the CHI (along with a visual examination of the clusters) provides a more objective measure and indicates HDBSCAN might reveal a more natural structure in the data.

Below are the UMAP and t-SNE visualizations of the clusters identified by HDBSCAN. As you can see,
there are three distinct clusters that indicate the contrastive learning worked:

<img alt="UMAP Visualization of HDBSCAN" height="400" src="assets/hdbscan-umap.png" width="500"/>
<img alt="t-SNE Visualization of HDBSCAN" height="400" src="assets/hdbscan-tsne.png" width="500"/>

Below are some example images from each cluster identified by HDBSCAN. There appears to be some
color similarity and perhaps slight structure/shape similarity among each cluster, which indicates
that perhaps the model has focused too much on color rather than structure. Although color may be a
useful tool for classifying galaxy morphology, it likely is not as important as structure and could
be a confounding variable:

<img alt="Example Images from Cluster Group 0" height="500" src="assets/group-0.png" width="500"/>
<img alt="Example Images from Cluster Group 1" height="500" src="assets/group-1.png" width="500"/>
<img alt="Example Images from Cluster Group 2" height="500" src="assets/group-2.png" width="500"/>

## Conclusion

**Outcomes:**

- Reconstructions were consistently really good
- Denoising was solid with some minor imperfections/artifacts
- Clustering/latent space representations were decent considering it did not have access to labeled
  data

I had initially hoped my clusters would more closely resemble the different galaxy morphology
classifications like the one below from another, more recent, Galaxy Zoo project named Galaxy Zoo:
DECaLS:

<img alt="Morphology Classifications from Galaxy Zoo: DECaLS" height="400" src="assets/gz_decals_morphology.png" width="800"/>

Although contrastive learning did help make the latent space representations more robust, this
proved to be quite challenging without introducing any labels. Additionally, these morphologies are
closely related and inherent qualities from each other, so it is often difficult to assign one
classification for each galaxy image. For example, see these images from the Galaxy Zoo: Hubble
project (Willett et al., 2016) visualizing the decision tree:

<img alt="Decision Tree Questions" height="800" src="assets/gz_hubble_decision_tree_questions.png" width="700"/>
<img alt="Decision Tree Diagram" height="600" src="assets/gz_hubble_decision_tree_diagram.png" width="600"/>

Although my initial goal was not met, my model still provides two useful tools while remaining
entirely unsupervised: cleaning images and clustering images. Additionally, depending on your
needs/goals, you could choose to use only one of these tools. By implementing some of the previously
mentioned changes or through further tuning, I believe this model could perform even better and
maybe even produce clusters that more closely resemble the morphology categories. Because only
self-supervision was used, this model could easily be adapted for use in any of the other projects
from Galaxy Zoo, such as GZ: 1, GZ: 2, GZ: CANDELS, GZ: DECaLS, and GZ: DESI.

## Acknowledgements

*U-Net: Convolutional Networks for Biomedical Image Segmentation*, Ronneberger et
al. [2015](https://doi.org/10.48550/arXiv.1505.04597)

*Galaxy Zoo: morphological classifications for 120,000 galaxies in HST legacy imaging*, Willet et
al. [2016](https://doi.org/10.48550/arXiv.1610.03068)

*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et
al. [2020](https://doi.org/10.48550/arXiv.2002.05709)

The [galaxy-datasets](https://github.com/mwalmsley/galaxy-datasets) GitHub repository created by Dr.
Mike Walmsley, a prominent researcher involved in the Galaxy Zoo project.

I acknowledge Research Advanced Computing Services (RACS) at the University of Oregon for providing
computing resources that have contributed to the research results reported within this project.
URL: https://racs.uoregon.edu.
