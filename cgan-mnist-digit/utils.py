import os,io
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython import display

"""
def plot_generated_images_tf(generator_model, digit_label, latent_dim=32, grid_dim=15,
                              dim1=0, dim2=1, save_path=None):
    """
    Plot a grid of generated images from a 2D slice of the latent space.

    Args:
        generator_model: The CGAN model instance with a .generate() method.
        digit_label (int): The digit class (0-9) to condition on.
        latent_dim (int): Dimensionality of latent space.
        grid_dim (int): Number of images per row and column in the grid.
        dim1 (int): First dimension index of latent vector to vary.
        dim2 (int): Second dimension index of latent vector to vary.
        save_path (str, optional): If provided, saves the image to the path.
    """
    digit_size = 28  # Assuming MNIST
    figure = np.zeros((digit_size * grid_dim, digit_size * grid_dim))
    grid_x = np.linspace(-4, 4, grid_dim)
    grid_y = np.linspace(-4, 4, grid_dim)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim), dtype=np.float32)
            z_sample[0, dim1] = xi
            z_sample[0, dim2] = yi

            label = tf.constant([digit_label], dtype=tf.int32)
            x_decoded = generator_model.generate(tf.convert_to_tensor(z_sample), label)
            x_decoded = x_decoded[0, :, :, 0].numpy()
            x_decoded = (x_decoded + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

            digit = x_decoded.reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title("CGAN: Images from Latent Space Grid", fontsize=16)
    plt.xlabel(f"z[{dim1}]")
    plt.ylabel(f"z[{dim2}]")
    plt.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved image to {save_path}")
    else:
        plt.show()

"""
def plot_generated_images_tf(generator_model, digit_label,
                             latent_dim=32, grid_dim=15,
                             dim1=0, dim2=1, save_path=None):
    digit_size = 28  # Assuming MNIST
    figure = np.zeros((digit_size * grid_dim, digit_size * grid_dim))
    grid_x = np.linspace(-4, 4, grid_dim)  # Create grid values in latent space (2D slice)
    grid_y = np.linspace(-4, 4, grid_dim)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim), dtype=np.float32)
            z_sample[0, dim1] = xi
            z_sample[0, dim2] = yi
            
            label = tf.constant([digit_label], dtype=tf.int32)
            x_decoded = generator_model.generate(tf.convert_to_tensor(z_sample), label)
            x_decoded = x_decoded[0, :, :, 0].numpy()
            x_decoded = (x_decoded + 1) / 2.0

            digit = x_decoded.reshape(digit_size, digit_size)
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title("CGAN: Images from Latent Space Grid", fontsize=16)
    plt.xlabel(f"z[{dim1}]")
    plt.ylabel(f"z[{dim2}]")
    plt.grid(False)

    if save_path:
        if isinstance(save_path, (str, bytes, os.PathLike)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved image to {save_path}")
        elif isinstance(save_path, io.BytesIO):
            plt.savefig(save_path, format="png", bbox_inches='tight')
        else:
            raise ValueError("save_path must be a file path or BytesIO")
        plt.close()
    else:
        plt.show()

def grad_norm(grads):
    """
    Compute global gradient norm from a list of gradients.

    Args:
        grads (list): List of gradient tensors.

    Returns:
        float: The total gradient norm.
    """
    total_norm = tf.norm([tf.norm(g) for g in grads if g is not None])
    return total_norm.numpy()


def compute_loss(model, real_images, real_output, fake_images, fake_output, labels,
                 lambda_gp=10.0, loss_type='vanilla'):
    """
    Compute generator and discriminator losses.

    Args:
        model: CGAN model with discriminate method.
        real_images: Batch of real images.
        real_output: Discriminator output for real images.
        fake_images: Batch of generated images.
        fake_output: Discriminator output for fake images.
        labels: Corresponding labels.
        lambda_gp (float): Weight for gradient penalty (WGAN only).
        loss_type (str): Either 'vanilla' or 'wgan'.

    Returns:
        Tuple[loss_gen, loss_disc]
    """
    if loss_type == 'vanilla':
        bce = keras.losses.BinaryCrossentropy(from_logits=False)

        disc_fake_loss = bce(tf.zeros_like(fake_output), fake_output)
        disc_real_loss = bce(tf.ones_like(real_output), real_output)
        loss_disc = disc_fake_loss + disc_real_loss

        loss_gen = bce(tf.ones_like(fake_output), fake_output)
        return loss_gen, loss_disc

    elif loss_type == 'wgan':
        loss_gen = -tf.reduce_mean(fake_output)
        loss_disc = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        # Gradient penalty
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images * alpha + fake_images * (1 - alpha)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = model.discriminate(interpolated, labels)

        grads = gp_tape.gradient(interpolated_output, interpolated)
        grads = tf.reshape(grads, [batch_size, -1])
        grad_norms = tf.norm(grads, axis=1)
        gp = tf.reduce_mean((grad_norms - 1.0) ** 2)

        loss_disc += lambda_gp * gp
        return loss_gen, loss_disc


def train_step(model, gen_optimizer, disc_optimizer, image_batch, label_batch,
               latent_dim, loss_type='vanilla', lambda_gp=10.0,
               clip_norm=False, max_norm=1.0):
    """
    Single training step for generator and discriminator.

    Args:
        model: CGAN model.
        gen_optimizer: Optimizer for generator.
        disc_optimizer: Optimizer for discriminator.
        image_batch: Batch of real images.
        label_batch: Corresponding labels.
        latent_dim: Dimension of latent vector.
        loss_type (str): 'vanilla' or 'wgan'.
        lambda_gp (float): Gradient penalty weight (for WGAN).
        clip_norm (bool): Whether to clip gradients.
        max_norm (float): Maximum gradient norm if clipping.

    Returns:
        Tuple of losses and gradient norms: (loss_gen, loss_disc, gen_grad_norm, disc_grad_norm)
    """
    assert loss_type in ["vanilla", "wgan"], "loss type must be either vanilla or wgan"

    batch_size = tf.shape(image_batch)[0]
    noise = tf.random.normal([batch_size, latent_dim])

    # --- Discriminator update ---
    with tf.GradientTape() as disc_tape:
        fake_images = model.generate(noise, label_batch)
        real_output = model.discriminate(image_batch, label_batch)
        fake_output = model.discriminate(fake_images, label_batch)

        loss_gen, loss_disc = compute_loss(model, image_batch, real_output,
                                           fake_images, fake_output, label_batch,
                                           lambda_gp=lambda_gp, loss_type=loss_type)

    disc_weights = model.disc_model.trainable_variables + model.embed_disc.trainable_variables
    disc_grads = disc_tape.gradient(loss_disc, disc_weights)
    if clip_norm:
        disc_grads, _ = tf.clip_by_global_norm(disc_grads, max_norm)
    disc_grad_norm = grad_norm(disc_grads)
    disc_optimizer.apply_gradients(zip(disc_grads, disc_weights))

    # --- Generator update ---
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape:
        fake_images = model.generate(noise, label_batch)
        fake_output = model.discriminate(fake_images, label_batch)

        loss_gen, _ = compute_loss(model, image_batch, real_output,
                                   fake_images, fake_output, label_batch,
                                   lambda_gp=lambda_gp, loss_type=loss_type)

    if hasattr(model, "latent_image"):
        gen_weights = (model.gen_model.trainable_variables +
                       model.embed_gen.trainable_variables +
                       model.latent_image.trainable_variables)
    else:
        gen_weights = model.gen_model.trainable_variables + model.embed_gen.trainable_variables

    gen_grads = gen_tape.gradient(loss_gen, gen_weights)
    if clip_norm:
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, max_norm)
    gen_grad_norm = grad_norm(gen_grads)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_weights))

    return loss_gen, loss_disc, gen_grad_norm, disc_grad_norm


def train_model(model, gen_optimizer, disc_optimizer, train_dataset, num_epochs,
                fixed_z, fixed_label, loss_type='vanilla', lambda_gp=10.0,
                clip_norm=False, max_norm=1.0):
    """
    Trains the CGAN model.

    Args:
        model: CGAN model.
        gen_optimizer: Optimizer for generator.
        disc_optimizer: Optimizer for discriminator.
        train_dataset: tf.data.Dataset of (image_batch, label_batch).
        num_epochs (int): Number of training epochs.
        fixed_z: Fixed latent vectors for visual monitoring.
        fixed_label: Fixed labels for image visualization.
        loss_type (str): 'vanilla' or 'wgan'.
        lambda_gp (float): Gradient penalty coefficient for WGAN.
        clip_norm (bool): Enable gradient clipping.
        max_norm (float): Max norm for gradient clipping.

    Returns:
        Tuple of lists: (gen_losses, disc_losses, gen_grad_norms, disc_grad_norms)
    """
    loss_file = open('loss.txt', 'w')
    gen_losses, disc_losses = [], []
    gen_grad_norms, disc_grad_norms = [], []

    for epoch in range(num_epochs):
        start = time.time()
        loss_gen = 0.0
        loss_disc = 0.0
        gen_grad_total = 0.0
        disc_grad_total = 0.0
        n = 0

        for image_batch, label_batch in train_dataset:
            loss_g, loss_d, gen_gn, disc_gn = train_step(
                model, gen_optimizer, disc_optimizer, image_batch, label_batch,
                latent_dim=fixed_z.shape[-1], loss_type=loss_type,
                lambda_gp=lambda_gp, clip_norm=clip_norm, max_norm=max_norm
            )
            loss_gen += loss_g
            loss_disc += loss_d
            gen_grad_total += gen_gn
            disc_grad_total += disc_gn
            if n % 100 == 0:
                print('.', end='')
            n += 1

        num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        gen_losses.append(float(loss_gen) / num_batches)
        disc_losses.append(float(loss_disc) / num_batches)
        gen_grad_norms.append(float(gen_grad_total) / num_batches)
        disc_grad_norms.append(float(disc_grad_total) / num_batches)

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            display.clear_output(wait=True)
            elapsed = (time.time() - start) / 60
            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} |'
                  f' time: {elapsed:.4f} min')
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}')

            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} |'
                  f' time: {elapsed:.4f} min', file=loss_file, flush=True)
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}',
                  file=loss_file, flush=True)

            model.generate_images(fixed_z, fixed_label)
            plt.savefig(f'generated_epoch_{epoch}.png')

    return gen_losses, disc_losses, gen_grad_norms, disc_grad_norms
