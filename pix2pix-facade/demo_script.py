import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Pix2Pix_Model
from dataset_preprocess import train_dataset, test_dataset
from utils import train_model, compute_metrics, evaluate_on_testset

# ------------------------ Reproducibility Setup ------------------------

seed = 32
keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

# ------------------------ Training Config ------------------------

NUM_EPOCHS = 200
LEARNING_RATE = 2e-4
BETA_1 = 0.5

# ------------------------ Model & Optimizers ------------------------

model_pix2pix = Pix2Pix_Model()
gen_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)
disc_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)

# ------------------------ Training ------------------------

gen_losses, disc_losses, gen_grad_norms, disc_grad_norms, gen_gan_losses, gen_l1_losses = \
    train_model(model_pix2pix, gen_optimizer, disc_optimizer,
                train_dataset, NUM_EPOCHS, clip_norm=False, max_norm=1.0)

# ------------------------ Metrics Logging ------------------------

df = pd.DataFrame({
    "gen_loss": gen_losses,
    "disc_loss": disc_losses,
    "gen_gan": gen_gan_losses,
    "gen_l1": gen_l1_losses,
    "gen_grad_norm": gen_grad_norms,
    "disc_grad_norm": disc_grad_norms
})

# ------------------------ Visualization: Gradient Norms ------------------------

df[['gen_grad_norm', 'disc_grad_norm']].plot(figsize=(8, 5), title="Gradient Norms")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("L2 Norm")
plt.show()

# ------------------------ Visualization: Generator & Discriminator Loss ------------------------

df[['gen_loss', 'disc_loss', 'gen_gan', 'gen_l1']].plot(figsize=(8, 5), title="Generator and Discriminator Loss")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ------------------------ Evaluation ------------------------

compute_metrics(model_pix2pix, test_dataset)
evaluate_on_testset(model_pix2pix, test_dataset, num_examples=10, save_images=False)
