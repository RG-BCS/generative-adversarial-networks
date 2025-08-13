import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from model import CGAN_MNIST_Digits_2
from utils import train_model, plot_generated_images_tf

# Set seeds and clear session for reproducibility
keras.backend.clear_session()
tf.random.set_seed(29)
np.random.seed(29)

# ---------------------
# Configuration
# ---------------------
NUM_EPOCHS = 10
BUFFER_SIZE = 60000 + 10000  # Full MNIST set (train + val)
BATCH_SIZE = 256
LATENT_DIM = 100
IMAGE_SHAPE = (28, 28, 1)
CLASS_LABELS = 10
LOSS_TYPE = 'vanilla'  # or 'wgan'
LEARNING_RATE = 1e-3

# ---------------------
# Load and preprocess data
# ---------------------
(train_images, train_labels), (valid_images, valid_labels) = keras.datasets.mnist.load_data()

# Normalize images to [-1, 1]
train_images = ((np.reshape(train_images, (-1, 28, 28, 1)).astype('float32')) / 255.0) * 2.0 - 1.0
valid_images = ((np.reshape(valid_images, (-1, 28, 28, 1)).astype('float32')) / 255.0) * 2.0 - 1.0

# Combine datasets
all_images = np.concatenate([train_images, valid_images], axis=0)
all_labels = np.concatenate([train_labels, valid_labels], axis=0)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# ---------------------
# Initialize model and optimizers
# ---------------------
model_cgan = CGAN_MNIST_Digits_2(
    latent_dim=LATENT_DIM,
    image_size=IMAGE_SHAPE,
    dropout=0.45,
    alpha=0.2,
    class_labels=CLASS_LABELS
)

gen_optimizer = keras.optimizers.Adam(2 * LEARNING_RATE, 0.5)
disc_optimizer = keras.optimizers.Adam(2 * LEARNING_RATE, 0.5)

# ---------------------
# Fixed latent space & labels for monitoring
# ---------------------
fixed_z = tf.random.normal(shape=[16, LATENT_DIM])
fixed_label = tf.random.uniform(shape=(16,), minval=0, maxval=10, dtype=tf.int32)

# ---------------------
# Train the CGAN
# ---------------------
gen_losses, disc_losses, gen_grad_norms, disc_grad_norms = train_model(
    model_cgan,
    gen_optimizer,
    disc_optimizer,
    train_dataset,
    NUM_EPOCHS,
    fixed_z,
    fixed_label,
    loss_type=LOSS_TYPE,
    lambda_gp=10.0,
    clip_norm=True,
    max_norm=5.0
)

# ---------------------
# Visualize loss and gradient norms
# ---------------------
df = pd.DataFrame({
    'gen_loss': gen_losses,
    'disc_loss': disc_losses,
    'gen_grad_norm': gen_grad_norms,
    'disc_grad_norm': disc_grad_norms
})

# Plot gradient norms
df[['gen_grad_norm', 'disc_grad_norm']].plot(
    figsize=(8, 5),
    title="Gradient Norms"
)
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("L2 Norm")
plt.show()

# Plot generator and discriminator losses
df[['gen_loss', 'disc_loss']].plot(
    figsize=(8, 5),
    title="Generator and Discriminator Loss"
)
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ---------------------
# Generate samples from different digits
# ---------------------
plot_generated_images_tf(model_cgan, digit_label=5, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1)
plot_generated_images_tf(model_cgan, digit_label=8, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1)
plot_generated_images_tf(model_cgan, digit_label=4, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1)
plot_generated_images_tf(model_cgan, digit_label=2, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1)
