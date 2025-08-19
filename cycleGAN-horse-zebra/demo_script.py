"""
dem_script.py

Run training and evaluation for CycleGAN on horse ↔ zebra dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---- Import your modules ----
from models import CycleGAN_Generator, CycleGAN_Discriminator
from utils import train_model, plot_training_metrics, evaluate_on_testset
from dataset import train_horses, train_zebras, test_horses, test_zebras

# ---- Reproducibility ----
seed = 42
keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

# ---- Hyperparameters ----
LAMBDA = 10
OUTPUT_CHANNELS = 3
NUM_EPOCHS = 20  # You can increase to 60+ for better results
LEARNING_RATE = 2e-4
BETA_1 = 0.5

# ---- Instantiate Models ----
gen_A2B = CycleGAN_Generator(input_image_size=(256, 256, 3), norm_type='instancenorm')
gen_B2A = CycleGAN_Generator(input_image_size=(256, 256, 3), norm_type='instancenorm')
disc_A = CycleGAN_Discriminator(norm_type='instancenorm')
disc_B = CycleGAN_Discriminator(norm_type='instancenorm')

# ---- Optimizers ----
gen_A2B_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)
gen_B2A_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)
disc_A_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)
disc_B_optimizer = keras.optimizers.Adam(LEARNING_RATE, BETA_1)

# ---- Train the model ----
print("Starting training...\n")
total_gen_1_loss, total_gen_2_loss, total_disc_1_loss, total_disc_2_loss, \
    gen_1_grad_norms, gen_2_grad_norms, disc_1_grad_norms, disc_2_grad_norms = train_model(
    gen_A2B, gen_B2A,
    gen_A2B_optimizer, gen_B2A_optimizer,
    disc_A, disc_B,
    disc_A_optimizer, disc_B_optimizer,
    train_horses, train_zebras,
    num_epochs=NUM_EPOCHS,
    clip_norm=False,
    max_norm=1.0,
    LAMBDA=LAMBDA
)

# ---- Collect training metrics ----
training_metrics = {
    "gen_A2B_loss": total_gen_1_loss,
    "gen_B2A_loss": total_gen_2_loss,
    "disc_A_loss": total_disc_1_loss,
    "disc_B_loss": total_disc_2_loss,
    "gen_A2B_grad_norm": gen_1_grad_norms,
    "gen_B2A_grad_norm": gen_2_grad_norms,
    "disc_A_grad_norm": disc_1_grad_norms,
    "disc_B_grad_norm": disc_2_grad_norms
}

# ---- Plot training curves ----
plot_training_metrics(training_metrics)

# ---- Evaluate on test set ----
print("\nTranslating test horses to zebras...")
evaluate_on_testset(gen_A2B, test_horses, direction='Horse to Zebra', save_images=True)

print("\nTranslating test zebras to horses...")
evaluate_on_testset(gen_B2A, test_zebras, direction='Zebra to Horse', save_images=True)
