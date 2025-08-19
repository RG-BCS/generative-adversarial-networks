"""
utils.py

Utility functions for training and evaluating the CycleGAN model,
including training steps, loss computations, evaluation plots,
and metric tracking.
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

# -----------------------------
# Evaluation Utilities
# -----------------------------

def evaluate_on_testset(generator_model, test_dataset, num_examples=5, direction='A2B', save_images=False):
    """
    Evaluate a generator model on a test dataset and visualize the output.
    """
    for idx, input_image in enumerate(test_dataset.take(num_examples)):
        print(f"Example {idx + 1}")
        prediction = generator_model(input_image, training=False)

        def denorm(img): return (img + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]

        input_image_np = denorm(input_image[0]).numpy()
        prediction_np = denorm(prediction[0]).numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image_np)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(prediction_np)
        plt.title(f"Translated ({direction})")
        plt.axis("off")

        if save_images:
            plt.savefig(f"test_result_{direction}_{idx+1:03d}.png")
        plt.show()


def plot_training_metrics(metrics_dict):
    """
    Plot generator/discriminator losses and gradient norms over training epochs.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(metrics_dict["gen_A2B_loss"], label="Gen A→B Loss")
    axs[0].plot(metrics_dict["gen_B2A_loss"], label="Gen B→A Loss")
    axs[0].plot(metrics_dict["disc_A_loss"], label="Disc A Loss")
    axs[0].plot(metrics_dict["disc_B_loss"], label="Disc B Loss")
    axs[0].set_title("Losses per Epoch")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(metrics_dict["gen_A2B_grad_norm"], label="Gen A→B Grad Norm")
    axs[1].plot(metrics_dict["gen_B2A_grad_norm"], label="Gen B→A Grad Norm")
    axs[1].plot(metrics_dict["disc_A_grad_norm"], label="Disc A Grad Norm")
    axs[1].plot(metrics_dict["disc_B_grad_norm"], label="Disc B Grad Norm")
    axs[1].set_title("Gradient Norms per Epoch")
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Training Utilities
# -----------------------------

def grad_norm(grads):
    """
    Compute the global gradient norm for a list of gradients.
    """
    return tf.linalg.global_norm([g for g in grads if g is not None])


def compute_loss(
    real_image_A, real_image_B,
    cycled_image_A, cycled_image_B,
    fake_disc_A, fake_disc_B,
    real_disc_A, real_disc_B,
    identity_genA2B_img_B, identity_genB2A_img_A,
    LAMBDA=10
):
    """
    Compute CycleGAN generator and discriminator losses.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Discriminator Losses
    fake_loss_A = loss_fn(tf.zeros_like(fake_disc_A), fake_disc_A)
    real_loss_A = loss_fn(tf.ones_like(real_disc_A), real_disc_A)
    total_disc_A_loss = (fake_loss_A + real_loss_A) * 0.5

    fake_loss_B = loss_fn(tf.zeros_like(fake_disc_B), fake_disc_B)
    real_loss_B = loss_fn(tf.ones_like(real_disc_B), real_disc_B)
    total_disc_B_loss = (fake_loss_B + real_loss_B) * 0.5

    # Generator Losses
    gan_A2B_loss = loss_fn(tf.ones_like(fake_disc_B), fake_disc_B)
    gan_B2A_loss = loss_fn(tf.ones_like(fake_disc_A), fake_disc_A)

    cycle_loss = tf.reduce_mean(tf.abs(real_image_A - cycled_image_A)) + \
                 tf.reduce_mean(tf.abs(real_image_B - cycled_image_B))

    identity_loss_A2B = tf.reduce_mean(tf.abs(identity_genA2B_img_B - real_image_B))
    identity_loss_B2A = tf.reduce_mean(tf.abs(identity_genB2A_img_A - real_image_A))

    total_gen_A2B_loss = gan_A2B_loss + LAMBDA * cycle_loss + identity_loss_A2B * LAMBDA * 0.5
    total_gen_B2A_loss = gan_B2A_loss + LAMBDA * cycle_loss + identity_loss_B2A * LAMBDA * 0.5

    return total_gen_A2B_loss, total_gen_B2A_loss, total_disc_A_loss, total_disc_B_loss


@tf.function
def train_step(
    gen_A2B, gen_B2A,
    gen_A2B_optimizer, gen_B2A_optimizer,
    disc_A, disc_B,
    disc_A_optimizer, disc_B_optimizer,
    image_A, image_B,
    clip_norm, max_norm, LAMBDA
):
    """
    Performs one training step for both generators and discriminators.
    """
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        generated_B = gen_A2B(image_A)
        generated_A = gen_B2A(image_B)

        cycled_A = gen_B2A(generated_B)
        cycled_B = gen_A2B(generated_A)

        identity_B = gen_A2B(image_B)
        identity_A = gen_B2A(image_A)

        real_disc_A = disc_A(image_A)
        fake_disc_A = disc_A(generated_A)

        real_disc_B = disc_B(image_B)
        fake_disc_B = disc_B(generated_B)

        # Compute loss
        total_gen_1_loss, total_gen_2_loss, total_disc_1_loss, total_disc_2_loss = compute_loss(
            image_A, image_B, cycled_A, cycled_B,
            fake_disc_A, fake_disc_B,
            real_disc_A, real_disc_B,
            identity_B, identity_A, LAMBDA
        )

    # Get gradients and apply optimizers
    def apply_gradients(optimizer, grads, weights):
        if clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, max_norm)
        optimizer.apply_gradients(zip(grads, weights))
        return grad_norm(grads)

    gen_1_grad_norm = apply_gradients(gen_A2B_optimizer, tape.gradient(total_gen_1_loss, gen_A2B.trainable_variables), gen_A2B.trainable_variables)
    gen_2_grad_norm = apply_gradients(gen_B2A_optimizer, tape.gradient(total_gen_2_loss, gen_B2A.trainable_variables), gen_B2A.trainable_variables)
    disc_1_grad_norm = apply_gradients(disc_A_optimizer, tape.gradient(total_disc_1_loss, disc_A.trainable_variables), disc_A.trainable_variables)
    disc_2_grad_norm = apply_gradients(disc_B_optimizer, tape.gradient(total_disc_2_loss, disc_B.trainable_variables), disc_B.trainable_variables)

    return total_gen_1_loss, total_gen_2_loss, total_disc_1_loss, total_disc_2_loss, \
           gen_1_grad_norm, gen_2_grad_norm, disc_1_grad_norm, disc_2_grad_norm


def train_model(
    gen_A2B, gen_B2A,
    gen_A2B_optimizer, gen_B2A_optimizer,
    disc_A, disc_B,
    disc_A_optimizer, disc_B_optimizer,
    train_dataset_A, train_dataset_B,
    num_epochs, clip_norm=False, max_norm=1.0, LAMBDA=10
):
    """
    Run full training loop for CycleGAN.
    """
    input_image_sample = next(iter(train_dataset_A))[0]
    target_image_sample = next(iter(train_dataset_B))[0]

    loss_file = open('loss.txt', 'w')

    # Track metrics
    total_gen_1_loss, total_gen_2_loss = [], []
    total_disc_1_loss, total_disc_2_loss = [], []
    gen_1_grad_norms, gen_2_grad_norms = [], []
    disc_1_grad_norms, disc_2_grad_norms = [], []

    start = time.time()
    for epoch in range(num_epochs):
        gen_1_loss, gen_2_loss = 0.0, 0.0
        disc_1_loss, disc_2_loss = 0.0, 0.0
        gen_1_grad, gen_2_grad = 0.0, 0.0
        disc_1_grad, disc_2_grad = 0.0, 0.0
        n = 0

        for (image_A, image_B) in tf.data.Dataset.zip((train_dataset_A, train_dataset_B)):
            step_results = train_step(
                gen_A2B, gen_B2A, gen_A2B_optimizer, gen_B2A_optimizer,
                disc_A, disc_B, disc_A_optimizer, disc_B_optimizer,
                image_A, image_B, clip_norm, max_norm, LAMBDA
            )

            gen_1_loss_ , gen_2_loss_, disc_1_loss_, disc_2_loss_, \
            gen_1_grad_, gen_2_grad_, disc_1_grad_, disc_2_grad_ = step_results

            gen_1_loss += gen_1_loss_
            gen_2_loss += gen_2_loss_
            disc_1_loss += disc_1_loss_
            disc_2_loss += disc_2_loss_
            gen_1_grad += gen_1_grad_
            gen_2_grad += gen_2_grad_
            disc_1_grad += disc_1_grad_
            disc_2_grad += disc_2_grad_

            if n % 100 == 0:
                print('.', end='')
            n += 1

        # Log epoch metrics
        num_batches = tf.data.experimental.cardinality(train_dataset_A).numpy()
        total_gen_1_loss.append(float(gen_1_loss) / num_batches)
        total_gen_2_loss.append(float(gen_2_loss) / num_batches)
        total_disc_1_loss.append(float(disc_1_loss) / num_batches)
        total_disc_2_loss.append(float(disc_2_loss) / num_batches)
        gen_1_grad_norms.append(float(gen_1_grad) / num_batches)
        gen_2_grad_norms.append(float(gen_2_grad) / num_batches)
        disc_1_grad_norms.append(float(disc_1_grad) / num_batches)
        disc_2_grad_norms.append(float(disc_2_grad) / num_batches)

        # Log and visualize
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            display.clear_output(wait=True)
            elapsed = (time.time() - start) / 60

            print(f'\nEpoch {epoch}/{num_epochs} | gen_1_loss: {total_gen_1_loss[-1]:.4f} |'
                  f' gen_2_loss: {total_gen_2_loss[-1]:.4f} | disc_1_loss: {total_disc_1_loss[-1]:.4f} |'
                  f' disc_2_loss: {total_disc_2_loss[-1]:.4f} | time: {elapsed:.2f} min')

            print(f' → Gradient norms — Gen_1: {gen_1_grad_norms[-1]:.4f} | Gen_2: {gen_2_grad_norms[-1]:.4f} |'
                  f' Disc_1: {disc_1_grad_norms[-1]:.4f} | Disc_2: {disc_2_grad_norms[-1]:.4f}')

            # Write to log file
            print(f'\nEpoch {epoch}/{num_epochs} | gen_1_loss: {total_gen_1_loss[-1]:.4f} |'
                  f' gen_2_loss: {total_gen_2_loss[-1]:.4f} | disc_1_loss: {total_disc_1_loss[-1]:.4f} |'
                  f' disc_2_loss: {total_disc_2_loss[-1]:.4f} | time: {elapsed:.2f} min', file=loss_file, flush=True)
            
            print(f'→ Gradient norms — Gen_1: {gen_1_grad_norms[-1]:.4f} | Gen_2: {gen_2_grad_norms[-1]:.4f} | '
                  f'Disc_1: {disc_1_grad_norms[-1]:.4f} | Disc_2: {disc_2_grad_norms[-1]:.4f}', file=loss_file, flush=True)

            # Display generated image sample
            input_image_ = (input_image_sample + 1) / 2.0
            target_image_ = (target_image_sample + 1) / 2.0
            generated_image = (gen_A2B(tf.expand_dims(input_image_sample, axis=0))[0] + 1) / 2.0

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(input_image_)
            axs[0].set_title('Input')
            axs[1].imshow(generated_image)
            axs[1].set_title('Generated')
            axs[2].imshow(target_image_)
            axs[2].set_title('Target')
            for ax in axs:
                ax.axis('off')
            plt.show()

            # Optionally save generated result
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                fig.savefig(f'cycleGAN_generated_epoch_{epoch}.png')

            plt.close(fig)
            start = time.time()

    return total_gen_1_loss, total_gen_2_loss, total_disc_1_loss, total_disc_2_loss, \
           gen_1_grad_norms, gen_2_grad_norms, disc_1_grad_norms, disc_2_grad_norms
