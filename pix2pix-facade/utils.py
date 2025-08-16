import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython import display
import tensorflow.image as tfi


# ------------------------ Loss & Metrics Utilities ------------------------

def grad_norm(grads):
    """
    Computes the global L2 norm of gradients.
    """
    return tf.linalg.global_norm([g for g in grads if g is not None])


def compute_loss(generated_image, real_image, fake_output, real_output, LAMBDA=100):
    """
    Computes generator and discriminator loss for Pix2Pix training.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Generator losses
    gan_loss = loss_fn(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_image - generated_image))
    total_gen_loss = gan_loss + LAMBDA * l1_loss

    # Discriminator losses
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    total_disc_loss = fake_loss + real_loss

    return total_gen_loss, total_disc_loss, gan_loss, l1_loss


def compute_metrics(model, test_dataset):
    """
    Computes average PSNR and SSIM over the test dataset.
    """
    psnr_scores, ssim_scores = [], []

    for input_image, target_image in test_dataset:
        prediction = model.generator(input_image)
        psnr = tfi.psnr(target_image, prediction, max_val=2.0)  # [-1, 1] range
        ssim = tfi.ssim(target_image, prediction, max_val=2.0)

        psnr_scores.append(psnr.numpy().mean())
        ssim_scores.append(ssim.numpy().mean())

    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")


def evaluate_on_testset(model, test_dataset, num_examples=5, save_images=False):
    """
    Displays predictions for a sample of images from the test set.
    """
    for idx, (input_image, target_image) in enumerate(test_dataset.take(num_examples)):
        print(f"Example {idx + 1}")
        model.generate_images(input_image, target_image)

        if save_images:
            plt.savefig(f"test_result_{idx+1:03d}.png")


# ------------------------ Training Functions ------------------------

@tf.function
def train_step(model, gen_optimizer, disc_optimizer,
               input_image_batch, target_image_batch,
               clip_norm=False, max_norm=1.0):
    """
    Single training step for one batch. Computes gradients and updates model weights.
    """
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        generated_images = model.generator(input_image_batch)
        real_output = model.discriminator(input_image_batch, target_image_batch)
        fake_output = model.discriminator(input_image_batch, generated_images)

        loss_gen, loss_disc, gen_gan_loss, gen_recon_loss = compute_loss(
            generated_images, target_image_batch,
            fake_output, real_output
        )

    # Discriminator updates
    disc_weights = model.disc_model.trainable_variables
    disc_grads = disc_tape.gradient(loss_disc, disc_weights)
    if clip_norm:
        disc_grads, _ = tf.clip_by_global_norm(disc_grads, max_norm)
    disc_grad_norm = grad_norm(disc_grads)
    disc_optimizer.apply_gradients(zip(disc_grads, disc_weights))

    # Generator updates
    gen_weights = (model.enc_gen.trainable_variables +
                   model.dec_gen.trainable_variables +
                   model.gen_last.trainable_variables)
    gen_grads = gen_tape.gradient(loss_gen, gen_weights)
    if clip_norm:
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, max_norm)
    gen_grad_norm = grad_norm(gen_grads)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_weights))

    return loss_gen, loss_disc, gen_grad_norm, disc_grad_norm, gen_gan_loss, gen_recon_loss


def train_model(model, gen_optimizer, disc_optimizer,
                train_dataset, num_epochs,
                clip_norm=False, max_norm=1.0):
    """
    Full training loop over all epochs.
    Logs, plots, and optionally saves intermediate outputs.
    """
    loss_file = open('loss.txt', 'w')
    gen_losses, gen_gan_losses, gen_l1_losses, disc_losses = [], [], [], []
    gen_grad_norms, disc_grad_norms = [], []

    for epoch in range(num_epochs):
        start = time.time()
        loss_gen, recon_loss, gan_gen_loss, loss_disc = 0.0, 0.0, 0.0, 0.0
        gen_grad_total, disc_grad_total = 0.0, 0.0
        n = 0

        for input_image_batch, target_image_batch in train_dataset:
            loss_g, loss_d, gen_gn, disc_gn, gen_gan, gen_recon = train_step(
                model, gen_optimizer, disc_optimizer,
                input_image_batch, target_image_batch,
                clip_norm=clip_norm, max_norm=max_norm
            )

            loss_gen += loss_g
            loss_disc += loss_d
            gen_grad_total += gen_gn
            disc_grad_total += disc_gn
            gan_gen_loss += gen_gan
            recon_loss += gen_recon

            if n % 100 == 0:
                print('.', end='')
            n += 1

        num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        gen_losses.append(float(loss_gen) / num_batches)
        disc_losses.append(float(loss_disc) / num_batches)
        gen_grad_norms.append(float(gen_grad_total) / num_batches)
        disc_grad_norms.append(float(disc_grad_total) / num_batches)
        gen_gan_losses.append(float(gan_gen_loss) / num_batches)
        gen_l1_losses.append(float(recon_loss) / num_batches)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            display.clear_output(wait=True)
            elapsed = (time.time() - start) / 60

            # Console logging
            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} |' \
                  f' gen_gan_loss: {gen_gan_losses[-1]:.4f} | gen_l1_loss: {gen_l1_losses[-1]:.4f} | time: {elapsed:.4f} min')
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}')

            # File logging
            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} |' \
                  f' gen_gan_loss: {gen_gan_losses[-1]:.4f} | gen_l1_loss: {gen_l1_losses[-1]:.4f} | time: {elapsed:.4f} min',
                  file=loss_file, flush=True)
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}',
                  file=loss_file, flush=True)

            # Show generated images during training
            input_image = (input_image_batch[0] + 1) / 2.0
            target_image = (target_image_batch[0] + 1) / 2.0
            generated_image = (model.generator(input_image_batch)[0] + 1) / 2.0

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(input_image)
            axs[0].set_title('Input')
            axs[1].imshow(generated_image)
            axs[1].set_title('Generated')
            axs[2].imshow(target_image)
            axs[2].set_title('Target')
            for ax in axs:
                ax.axis('off')
            plt.show()

            # Save visual output occasionally
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                fig.savefig(f'generated_epoch_{epoch}.png')
            plt.close(fig)

    return gen_losses, disc_losses, gen_grad_norms, disc_grad_norms, gen_gan_losses, gen_l1_losses
