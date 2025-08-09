import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_wgan_with_gp_model, train_wgan_with_gp_model_2, show_super_resolution
from dataloader_generator import train_dl, valid_dl
from model import WGAN_GP_Div2k, WGAN_GP_Div2k_2  # Assuming your models are defined here
from device import device  # Assuming you have a device.py defining device = torch.device(...) or define it here

# Set device if not imported
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 43
torch.manual_seed(seed)

num_epochs = 100
lambda_gp = 10
n_filters = 128
learning_rate = 1e-3

# Initialize model and send to device
# model_res = WGAN_GP_Div2k(n_filters=n_filters).to(device)
model_res = WGAN_GP_Div2k_2(n_filters=n_filters, dropout=0.1, negative_slope=0.2, skip_connect=True).to(device)

# Separate optimizers for generator and discriminator
gen_optim_res = torch.optim.Adam(model_res.gen_model.parameters(), lr=learning_rate)
disc_optim_res = torch.optim.Adam(model_res.disc_model.parameters(), lr=learning_rate)

# Train model
gen_train_loss, disc_train_loss, gen_val_loss, disc_val_loss = train_wgan_with_gp_model_2(
    model_res, gen_optim_res, disc_optim_res, num_epochs, train_dl, valid_dl,
    lambda_gp=lambda_gp,
    clip_norm_disc=True,
    clip_norm_gen=True,
    norm_max_disc=20.0,
    norm_max_gen=20.0,
    n_critic=2,
    apply_recon_loss=True,
    recon_loss_weight=200.0,
    apply_noise_to_real=True
)

# Plot training history
history = {
    'gen_train_loss': gen_train_loss,
    'disc_train_loss': disc_train_loss,
    'gen_val_loss': gen_val_loss,
    'disc_val_loss': disc_val_loss
}
pd.DataFrame(history).plot()
plt.grid()
plt.show()

print("Sample images from train set after model is trained")
show_super_resolution(model_res, dataloader=train_dl, num_images=8, calculate_quantitative=False, random_sample=True)

print("\nSample images from valid set after model is trained")
show_super_resolution(model_res, dataloader=valid_dl, num_images=8, calculate_quantitative=False, random_sample=True)
