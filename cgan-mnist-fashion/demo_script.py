# demo_script.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from utils import train_model, plot_generated_images_tf
from models import CGAN_MNIST_Fashion

# --- Reproducibility ---
keras.backend.clear_session()
tf.random.set_seed(21)
np.random.seed(21)

# --- Hyperparameters ---
NUM_EPOCHS = 30
BUFFER_SIZE = 60000 + 10000
BATCH_SIZE = 256
LATENT_DIM = 100
NUM_FEATURE_MAP = 256
IMAGE_SHAPE = (28, 28, 1)
CLASS_LABELS = 10
LOSS_TYPE = 'vanilla'  # Options: 'vanilla' or 'wgan'
LEARNING_RATE = 2e-3

# --- Load Fashion-MNIST Dataset ---
(train_images, train_labels), (valid_images, valid_labels) = keras.datasets.fashion_mnist.load_data()

train_images = ((np.reshape(train_images, (-1, 28, 28, 1)).astype('float32')) / 255.0) * 2.0 - 1.0
valid_images = ((np.reshape(valid_images, (-1, 28, 28, 1)).astype('float32')) / 255.0) * 2.0 - 1.0

all_images = np.concatenate([train_images, valid_images], axis=0)
all_labels = np.concatenate([train_labels, valid_labels], axis=0)

# --- Create Dataset Pipeline ---
train_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# --- Instantiate CGAN ---
model_cgan = CGAN_MNIST_Fashion(latent_dim=LATENT_DIM,
                                 num_feature_map=NUM_FEATURE_MAP,
                                 dropout=0.45,
                                 alpha=0.2,
                                 image_size=IMAGE_SHAPE,
                                 class_labels=CLASS_LABELS)

# --- Optimizers ---
gen_optimizer = keras.optimizers.Adam(LEARNING_RATE, 0.5)
disc_optimizer = keras.optimizers.Adam(LEARNING_RATE, 0.5)

# --- Fixed noise & labels for visualization ---
fixed_z = tf.random.normal(shape=[16, LATENT_DIM])
fixed_label = tf.random.uniform(shape=(16,), minval=0, maxval=10, dtype=tf.int32)

# --- Train the model ---
gen_losses, disc_losses, gen_grad_norms, disc_grad_norms = train_model(
    model=model_cgan,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    train_dataset=train_dataset,
    num_epochs=NUM_EPOCHS,
    fixed_z=fixed_z,
    fixed_label=fixed_label,
    loss_type=LOSS_TYPE,
    lambda_gp=10.0,
    clip_norm=True,
    max_norm=5.0
)

# --- Plot Training Statistics ---
df = pd.DataFrame({
    'gen_loss': gen_losses,
    'disc_loss': disc_losses,
    'gen_grad_norm': gen_grad_norms,
    'disc_grad_norm': disc_grad_norms
})

# Plot gradient norms
df[['gen_grad_norm', 'disc_grad_norm']].plot(figsize=(8, 5), title="Gradient Norms")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("L2 Norm")
plt.tight_layout()
plt.savefig("gradient_norms.png")
plt.close()

# Plot generator and discriminator loss
df[['gen_loss', 'disc_loss']].plot(figsize=(8, 5), title="Generator and Discriminator Loss")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("training_loss.png")
plt.close()

# --- Visualize Latent Space Projections ---
for label in [1, 3, 7, 9]:
    plot_generated_images_tf(model_cgan,
                              digit_label=label,
                              latent_dim=LATENT_DIM,
                              grid_dim=3,
                              dim1=0,
                              dim2=1,
                              save_path=f"latent_projection_class_{label}.png")
