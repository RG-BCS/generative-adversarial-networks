# utils.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def plot_generated_images_tf(generator_model, digit_label, latent_dim=32, grid_dim=15,
                              dim1=0, dim2=1, save_path=None):
    """
    Plots a grid of generated images by varying two latent dimensions,
    conditioned on a specific class label.
    """
    digit_size = 28
    figure = np.zeros((digit_size * grid_dim, digit_size * grid_dim))
    grid_x = np.linspace(-4, 4, grid_dim)
    grid_y = np.linspace(-4, 4, grid_dim)[::-1]

    CLASS_NAMES = {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
    }

    class_name = CLASS_NAMES.get(digit_label, f"Class {digit_label}")

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
    plt.title(f"CGAN Generated: {class_name} Images", fontsize=16)
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


def grad_norm(grads):
    """
    Computes the global gradient norm for monitoring.
    """
    total_norm = tf.norm([tf.norm(g) for g in grads if g is not None])
    return total_norm.numpy()


def compute_loss(model, real_images, real_output, fake_images, fake_output, labels, lambda_gp=10.0, loss_type='vanilla'):
    """
    Computes generator and discriminator loss for CGAN.
    
    Args:
        model: CGAN model
        real_images: Real input images
        real_output: Discriminator output on real images
        fake_images: Generated images
        fake_output: Discriminator output on fake images
        labels: Class labels
        lambda_gp: Gradient penalty term (WGAN only)
        loss_type: 'vanilla' or 'wgan'
    
    Returns:
        Tuple of (generator_loss, discriminator_loss)
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


def train_step(model, gen_optimizer, disc_optimizer, image_batch, label_batch, latent_dim, 
               loss_type='vanilla', lambda_gp=10.0, clip_norm=False, max_norm=1.0):
    """
    Single training step for both discriminator and generator.
    """
    assert loss_type in ["vanilla", "wgan"], "loss_type must be 'vanilla' or 'wgan'"

    batch_size = tf.shape(image_batch)[0]
    noise = tf.random.normal([batch_size, latent_dim])

    # --- Discriminator Step ---
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

    # --- Generator Step ---
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape:
        fake_images = model.generate(noise, label_batch)
        fake_output = model.discriminate(fake_images, label_batch)

        loss_gen, _ = compute_loss(model, image_batch, real_output,
                                   fake_images, fake_output, label_batch,
                                   lambda_gp=lambda_gp, loss_type=loss_type)

    gen_weights = (model.gen_model.trainable_variables +
                   model.embed_gen.trainable_variables +
                   model.latent_image.trainable_variables)

    gen_grads = gen_tape.gradient(loss_gen, gen_weights)
    if clip_norm:
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, max_norm)
    gen_grad_norm = grad_norm(gen_grads)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_weights))

    return loss_gen, loss_disc, gen_grad_norm, disc_grad_norm


def train_model(model, gen_optimizer, disc_optimizer, train_dataset, num_epochs, fixed_z, fixed_label,
                loss_type='vanilla', lambda_gp=10.0, clip_norm=False, max_norm=1.0):
    """
    Full training loop for CGAN.
    Logs loss, gradient norms, and optionally displays or saves generated samples.
    """
    loss_file = open('loss.txt', 'w')
    gen_losses, disc_losses = [], []
    gen_grad_norms, disc_grad_norms = [], []

    for epoch in range(num_epochs):
        loss_gen, loss_disc = 0.0, 0.0
        gen_grad_total, disc_grad_total = 0.0, 0.0
        n = 0

        for image_batch, label_batch in train_dataset:
            loss_g, loss_d, gen_gn, disc_gn = train_step(model, gen_optimizer, disc_optimizer,
                                                         image_batch, label_batch,
                                                         latent_dim=fixed_z.shape[-1],
                                                         loss_type=loss_type,
                                                         lambda_gp=lambda_gp,
                                                         clip_norm=clip_norm,
                                                         max_norm=max_norm)
            loss_gen += loss_g
            loss_disc += loss_d
            gen_grad_total += gen_gn
            disc_grad_total += disc_gn
            n += 1
            if n % 100 == 0:
                print('.', end='')

        num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        gen_losses.append(float(loss_gen) / num_batches)
        disc_losses.append(float(loss_disc) / num_batches)
        gen_grad_norms.append(float(gen_grad_total) / num_batches)
        disc_grad_norms.append(float(disc_grad_total) / num_batches)

        # Logging
        print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f}')
        print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}')
        print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f}', file=loss_file, flush=True)
        print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}', file=loss_file, flush=True)

        # Generate and save images periodically
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.generate_images(fixed_z, fixed_label)

        if epoch % 8 == 0 or epoch == num_epochs - 1:
            pred = model.generate(fixed_z, fixed_label)
            fig = plt.figure(figsize=(4, 4))
            for i in range(pred.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow((pred[i, :, :, 0] + 1) / 2.0, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            fig.savefig(f'generated_fashion_{epoch}.png')
            plt.close(fig)

    return gen_losses, disc_losses, gen_grad_norms, disc_grad_norms
