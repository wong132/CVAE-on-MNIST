# CVAE-on-MNIST
This repository contains an implementation of a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset. The project utilizes PyTorch to build and train the model.

Overview

A Conditional Variational Autoencoder (CVAE) is a type of generative model that learns a latent representation of input data conditioned on a label. In this project, the CVAE is trained on the MNIST dataset to generate handwritten digits conditioned on class labels.

Dataset

The MNIST dataset, consisting of 70,000 grayscale images of handwritten digits (0-9), is used for training and evaluation. The dataset is loaded using torchvision.datasets.MNIST and is preprocessed before being fed into the model.

Implementation

1. Data Loading & Preprocessing

The dataset is loaded using torchvision.datasets.MNIST.

Images are transformed into tensors and normalized.

Labels are one-hot encoded for conditioning.

2. Model Architecture

The CVAE consists of:

Encoder: Maps input images and labels to a latent space representation.

Latent Space: Uses a reparameterization trick to sample from a learned distribution.

Decoder: Generates images from latent vectors and labels.

3. Training

The model is trained using the Adam optimizer and a reconstruction loss (Binary Cross-Entropy) combined with the KL divergence loss.

A batch size of 32 and learning rate of 0.003 are used.

Running the Code

To train and evaluate the model, run the notebook in a Jupyter Notebook or Python environment:

4. Results

The trained CVAE can generate images based on the given class labels.

The performance of the model can be visualized using Matplotlib.

5. Evaluation

We use the Fréchet Inception Distance (FID) score to evaluate image quality. A lower FID score indicates better similarity between generated and real images. It is computed using feature statistics (mean & covariance) from a pre-trained Inception network. The formula for FID score is:

\[
FID = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
\]

where:  
- \( \mu_r, \Sigma_r \) = Mean and covariance of real images  
- \( \mu_g, \Sigma_g \) = Mean and covariance of generated images  
