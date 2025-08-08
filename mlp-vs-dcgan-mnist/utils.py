import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

# Ensure consistent device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_noise(batch_size, z_size, z_mode):
    """
    Generate random noise for MLP-based GAN (1D latent vectors).

    Args:
        batch_size (int): Number of samples.
        z_size (int): Latent vector size.
        z_mode (str): 'uniform' or 'normal'

    Returns:
        Tensor: Random noise tensor of shape (batch_size, z_size)
    """
    if z_mode == 'uniform':
        return torch.rand(batch_size, z_size) * 2 - 1
    elif z_mode == 'normal':
        return torch.randn(batch_size, z_size)


def dcgan_random_noise(batch_size, z_size, mode_z):
    """
    Generate random noise for DCGAN (4D latent vectors for ConvTranspose2d).

    Args:
        batch_size (int): Number of samples.
        z_size (int): Latent vector depth.
        mode_z (str): 'uniform' or 'normal'

    Returns:
        Tensor: Random noise of shape (batch_size, z_size, 1, 1)
    """
    if mode_z == 'uniform':
        return torch.rand(batch_size, z_size, 1, 1) * 2 - 1
    elif mode_z == 'normal':
        return torch.randn(batch_size, z_size, 1, 1)


def create_samples(model, input_z, image_size=(28, 28)):
    """
    Generate and normalize sample images from generator.

    Args:
        model (nn.Module): Generator model (callable)
        input_z (Tensor): Latent input vectors
        image_size (tuple): Shape of the output image (default: (28, 28))

    Returns:
        Tensor: Normalized image tensor in range [0, 1]
    """
    predictions = model(input_z).reshape(input_z.size(0), *image_size)
    return (predictions + 1) / 2.0


def plot_generated_images(decoder, grid_dim=15, latent_dim=32, dim1=0, dim2=1, model_type='nn_gan'):
    """
    Create and display a grid of generated images by sampling a 2D latent space.

    Args:
        decoder (nn.Module): Generator model
        grid_dim (int): Number of rows and columns in the grid
        latent_dim (int): Dimensionality of latent vector
        dim1 (int): First latent dimension to vary across x-axis
        dim2 (int): Second latent dimension to vary across y-axis
        model_type (str): 'nn_gan' or 'dc_gan'
    """
    digit_size = 28  # For MNIST
    figure = np.zeros((digit_size * grid_dim, digit_size * grid_dim))
    grid_x = np.linspace(-4, 4, grid_dim)
    grid_y = np.linspace(-4, 4, grid_dim)[::-1]  # Top to bottom

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if model_type == 'nn_gan':
                z_sample = torch.randn(1, latent_dim, device=device)  # MLP-style latent vector
            else:
                z_sample = torch.randn(1, latent_dim, 1, 1, device=device)  # DCGAN-style

            x_decoded = decoder(z_sample)  # Generate image

            digit = x_decoded[0].reshape(digit_size, digit_size).detach().cpu().numpy()
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = digit

    plt.figure(figsize=(15, 12))
    title = "MLP-GAN: Images Generated After Training" if model_type == 'nn_gan' else "CNN-GAN: Images Generated After Training"
    plt.title(title, fontsize=20)
    plt.xlabel(f"z[{dim1}]")
    plt.ylabel(f"z[{dim2}]")
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    plt.show()


def grad_norm(model):
    """
    Compute the total L2 norm of the gradients of all parameters in a model.

    Args:
        model (nn.Module): The model whose gradients will be computed

    Returns:
        float: Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_gan_model(model, gen_optimizer, disc_optimizer, loss_fn, num_epochs, train_dl,
                    latent_dim, training_progress_check, fixed_z, model_type='nn_gan'):
    """
    Train a GAN model (MLP or DCGAN) with specified optimizers and data.

    Args:
        model (nn.Module): GAN model containing both generator and discriminator
        gen_optimizer (Optimizer): Optimizer for generator
        disc_optimizer (Optimizer): Optimizer for discriminator
        loss_fn (Loss): Loss function (e.g., BCEWithLogitsLoss)
        num_epochs (int): Number of epochs to train
        train_dl (DataLoader): Training data loader
        latent_dim (int): Size of latent noise vector
        training_progress_check (Tensor): Latent noise to visualize generation progress
        fixed_z (Tensor): Latent noise used to generate evaluation samples
        model_type (str): 'nn_gan' or 'dc_gan'

    Returns:
        Tuple[List[float], List[float], List[np.ndarray]]:
            - Generator loss history
            - Discriminator loss history
            - Epoch-wise generated samples
    """
    assert model_type in ['nn_gan', 'dc_gan'], "Unsupported model type."
    epoch_samples = []
    gen_losses, disc_losses = [], []
    start = time.time()

    for epoch in range(num_epochs):
        g_loss, d_loss = 0.0, 0.0
        n = 0

        for real_data, _ in train_dl:
            batch_size = real_data.size(0)

            # Reshape real data
            if model_type == 'nn_gan':
                real_data = real_data.view(batch_size, -1).to(device)
            else:
                real_data = real_data.to(device)

            # === Train Discriminator ===
            disc_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            real_preds = model.discriminate(real_data)
            real_loss = loss_fn(real_preds, real_labels)

            # Generate fake data
            if model_type == 'nn_gan':
                z = torch.randn(batch_size, latent_dim, device=device)
            else:
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_data = model.generate(z)

            fake_preds = model.discriminate(fake_data.detach())
            fake_loss = loss_fn(fake_preds, fake_labels)

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_grad_norm = grad_norm(model.disc_model)
            disc_optimizer.step()
            d_loss += disc_loss

            # === Train Generator ===
            gen_optimizer.zero_grad()
            if model_type == 'nn_gan':
                z = torch.randn(batch_size, latent_dim, device=device)
            else:
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_data = model.generate(z)
            gen_preds = model.discriminate(fake_data)
            gen_loss = loss_fn(gen_preds, real_labels)
            gen_loss.backward()
            gen_grad_norm = grad_norm(model.gen_model)
            gen_optimizer.step()
            g_loss += gen_loss

            if n % 100 == 0:
                print('.', end='')
            n += 1

        # Average losses
        d_loss /= len(train_dl)
        g_loss /= len(train_dl)
        gen_losses.append(g_loss.item())
        disc_losses.append(d_loss.item())

        # Save generated samples from fixed_z
        epoch_samples.append(create_samples(model.generate, fixed_z).detach().cpu().numpy())

        # Visualization
        display.clear_output(wait=True)
        model.generate_images(training_progress_check)
        elapsed = (time.time() - start) / 60

        print(f"Epoch [{epoch+1}/{num_epochs}] | Disc_Loss: {disc_loss.item():.4f} | Genr_Loss: {gen_loss.item():.4f} "
              f"| dis_grad: {disc_grad_norm:.4f} | gen_grad: {gen_grad_norm:.4f} | elapsed_time: {elapsed:.4f} min")

        start = time.time()

    return gen_losses, disc_losses, epoch_samples
