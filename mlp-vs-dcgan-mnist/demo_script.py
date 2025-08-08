import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from models import GAN_NN_MNIST, DCGAN_MNIST
from utils import random_noise, dcgan_random_noise, train_gan_model, plot_generated_images

# Set random seed for reproducibility
seed = 10
torch.manual_seed(seed)

# Hyperparameters for MLP-GAN
num_epochs = 50
batch_size = 32
latent_dim = 100
z_mode = 'uniform'
image_size = (28, 28)
input_channel = 1
n_filters = 32
hidden_units = 256
num_layers = 3

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
train_dl = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize MLP-GAN model
model_nn = GAN_NN_MNIST(input_size=latent_dim, gen_hidden_units=hidden_units, gen_num_layers=num_layers, 
                        gen_output_size=784, disc_hidden_units=hidden_units, disc_num_layers=num_layers,
                        disc_output_size=1, dropout=0.1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Loss and optimizers
loss_fn = nn.BCEWithLogitsLoss()
gen_optimizer_nn = torch.optim.Adam(model_nn.gen_model.parameters(), lr=1e-3)
disc_optimizer_nn = torch.optim.Adam(model_nn.disc_model.parameters(), lr=1e-3)

# Generate fixed noise vectors for evaluation
fixed_z = random_noise(batch_size, latent_dim, z_mode).to(model_nn.gen_model[0].weight.device)
training_progress_check = random_noise(16, latent_dim, z_mode)

# Train MLP-GAN
all_g_losses, all_d_losses, epoch_samples = train_gan_model(
    model_nn, gen_optimizer_nn, disc_optimizer_nn, loss_fn,
    num_epochs, train_dl, latent_dim, training_progress_check,
    fixed_z, model_type='nn_gan'
)

# Plot generator and discriminator losses
g_losses_cpu = [g.cpu() if torch.is_tensor(g) else g for g in all_g_losses]
half_d_losses = [d.cpu() / 2 if torch.is_tensor(d) else d / 2 for d in all_d_losses]
plt.plot(g_losses_cpu, label='Generator loss')
plt.plot(half_d_losses, label='Discriminator loss')
plt.legend(fontsize=20)
plt.xlabel('Epoch', size=15)
plt.ylabel('Loss', size=15)
plt.title("MLP-GAN")
plt.grid()
plt.show()

# Visualize generated samples across selected epochs
selected_epochs = [1, 2, 25, 42, 48, num_epochs]
fig = plt.figure(figsize=(10, 14))
plt.suptitle("MLP-GAN: Sample Images Created During Training", fontsize=12, color='blue', y=0.92)

for i, e in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i*5 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.06, 0.5, f'Epoch {e}', rotation=90, size=18, color='red',
                    horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        image = epoch_samples[e - 1][j]
        ax.imshow(image, cmap='gray_r')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Grid of generated samples
plot_generated_images(model_nn.gen_model, grid_dim=20, latent_dim=latent_dim, model_type='nn_gan')

# ----------------------- CNN GAN -----------------------

# Updated hyperparams for DCGAN
num_epochs = 50
latent_dim = 100
z_mode = 'uniform'
image_size = (28, 28)
n_filters = 64
batch_size = 32

# Reload dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
train_dl = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize DCGAN model
model_cnn = DCGAN_MNIST(latent_dim, n_filters, dropout=0.1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Loss and optimizers
loss_fn = nn.BCEWithLogitsLoss()
gen_optimizer_dc = torch.optim.Adam(model_cnn.gen_model.parameters(), lr=1e-3)
disc_optimizer_dc = torch.optim.Adam(model_cnn.disc_model.parameters(), lr=1e-3)

# Generate fixed noise vectors for evaluation
fixed_z = dcgan_random_noise(batch_size, latent_dim, z_mode).to(model_cnn.gen_model[0].weight.device)
training_progress_check = dcgan_random_noise(16, latent_dim, z_mode)

# Train DCGAN
all_g_loss_cnn, all_d_loss_cnn, epoch_samples_cnn = train_gan_model(
    model_cnn, gen_optimizer_dc, disc_optimizer_dc,
    loss_fn, num_epochs, train_dl, latent_dim,
    training_progress_check, fixed_z, model_type='dc_gan'
)

# Plot generator and discriminator losses
g_losses_cpu_cnn = [g.cpu() if torch.is_tensor(g) else g for g in all_g_loss_cnn]
half_d_loss_cnn = [d.cpu() / 2 if torch.is_tensor(d) else d / 2 for d in all_d_loss_cnn]
plt.plot(g_losses_cpu_cnn, label='Generator loss')
plt.plot(half_d_loss_cnn, label='Discriminator loss')
plt.legend(fontsize=20)
plt.title("CNN-GAN")
plt.xlabel('Epoch', size=15)
plt.ylabel('Loss', size=15)
plt.grid()
plt.show()

# Visualize generated samples from DCGAN
selected_epochs = [1, 2, 25, 42, 48, num_epochs]
fig = plt.figure(figsize=(10, 14))
plt.suptitle("CNN-GAN: sample images created during training", fontsize=12, color='blue', y=0.92)

for i, e in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i*5 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.06, 0.5, f'Epoch {e}', rotation=90, size=18, color='red',
                    horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        image = epoch_samples_cnn[e - 1][j]
        ax.imshow(image, cmap='gray_r')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Final DCGAN grid output
plot_generated_images(model_cnn.gen_model, grid_dim=20, latent_dim=latent_dim, model_type='cnn_gan')
