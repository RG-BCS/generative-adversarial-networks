import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd

# ---------------- Model Definitions -----------------

class CGAN_MNIST_Digits(keras.Model):
    def __init__(self, latent_dim=32, image_size=(28, 28, 1), class_labels=10,
                 alpha=0.2, momentum=0.8, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.class_labels = class_labels

        self.embed_gen = keras.layers.Embedding(input_dim=class_labels, output_dim=latent_dim)
        self.embed_disc = keras.layers.Embedding(input_dim=class_labels, output_dim=np.prod(image_size))

        self.gen_model = keras.Sequential([
            keras.layers.InputLayer(shape=(latent_dim,)),
            keras.layers.Dense(7 * 7 * 128, use_bias=False),
            keras.layers.BatchNormalization(momentum=momentum),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Reshape((7, 7, 128)),

            keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False),
            keras.layers.BatchNormalization(momentum=momentum),
            keras.layers.LeakyReLU(alpha),

            keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same',
                                         use_bias=False, activation='tanh')
        ])

        self.disc_model = keras.Sequential([
            keras.layers.InputLayer(shape=image_size),
            keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def generate(self, z, label):
        cond = self.embed_gen(label)
        z_cond = tf.multiply(z, cond)
        return self.gen_model(z_cond)

    def discriminate(self, image, label):
        cond = tf.reshape(self.embed_disc(label), (-1, *self.image_size))
        image_cond = tf.multiply(image, cond)
        return self.disc_model(image_cond)

    def generate_images(self, test_sample, rand_label):
        pred = self.generate(test_sample, rand_label)
        fig = plt.figure(figsize=(4, 4))
        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((pred[i, :, :, 0] + 1) / 2.0, cmap='gray')
            plt.axis('off')
        plt.show()


# ---------------- Utility Functions -----------------

def plot_generated_images_tf(generator_model, digit_label, latent_dim=32, grid_dim=15, dim1=0, dim2=1, save_path=None):
    import matplotlib
    matplotlib.use('Agg')

    digit_size = 28
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved image to {save_path}")
    else:
        plt.show()


def grad_norm(grads):
    total_norm = tf.norm([tf.norm(g) for g in grads if g is not None])
    return total_norm.numpy()


def compute_loss(model, real_images, real_output, fake_images, fake_output, labels, lambda_gp=10.0, loss_type='vanilla'):
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

    assert loss_type in ["vanilla", "wgan"], "loss type must be either vanilla or wgan"

    batch_size = tf.shape(image_batch)[0]
    noise = tf.random.normal([batch_size, latent_dim])

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

    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape:
        fake_images = model.generate(noise, label_batch)
        fake_output = model.discriminate(fake_images, label_batch)

        loss_gen, _ = compute_loss(model, image_batch, real_output,
                                   fake_images, fake_output, label_batch,
                                   lambda_gp=lambda_gp, loss_type=loss_type)

    gen_weights = model.gen_model.trainable_variables + model.embed_gen.trainable_variables
    gen_grads = gen_tape.gradient(loss_gen, gen_weights)
    if clip_norm:
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, max_norm)
    gen_grad_norm = grad_norm(gen_grads)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_weights))

    return loss_gen, loss_disc, gen_grad_norm, disc_grad_norm


def train_model(model, gen_optimizer, disc_optimizer, train_dataset, num_epochs, fixed_z, fixed_label,
                loss_type='vanilla', lambda_gp=10.0, clip_norm=False, max_norm=1.0):
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
            loss_g, loss_d, gen_gn, disc_gn = train_step(model, gen_optimizer, disc_optimizer, image_batch, label_batch,
                                                         latent_dim=fixed_z.shape[-1], loss_type=loss_type,
                                                         lambda_gp=lambda_gp, clip_norm=clip_norm, max_norm=max_norm)
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
            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} | time: {elapsed:.4f} min')
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}')
            print(f'\nEpoch {epoch}/{num_epochs} | gen_loss: {gen_losses[-1]:.4f} | disc_loss: {disc_losses[-1]:.4f} | time: {elapsed:.4f} min', file=loss_file, flush=True)
            print(f' → Gradient norms — Gen: {gen_grad_norms[-1]:.4f}, Disc: {disc_grad_norms[-1]:.4f}', file=loss_file, flush=True)

            model.generate_images(fixed_z, fixed_label)

    loss_file.close()
    return gen_losses, disc_losses, gen_grad_norms, disc_grad_norms


# ---------------- Main Script -----------------

if __name__ == '__main__':
    keras.backend.clear_session()
    tf.random.set_seed(29)
    np.random.seed(29)

    NUM_EPOCHS = 30
    BUFFER_SIZE = 70000
    BATCH_SIZE = 100
    LATENT_DIM = 128
    IMAGE_SHAPE = (28, 28, 1)
    CLASS_LABELS = 10
    LOSS_TYPE = 'vanilla'  # or 'wgan'
    LEARNING_RATE = 1e-3

    (train_images, train_labels), (valid_images, valid_labels) = keras.datasets.mnist.load_data()
    train_images = (np.reshape(train_images, (-1, 28, 28, 1)).astype('float32') - 127.5) / 127.5
    valid_images = (np.reshape(valid_images, (-1, 28, 28, 1)).astype('float32') - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = CGAN_MNIST_Digits(latent_dim=LATENT_DIM, image_size=IMAGE_SHAPE, class_labels=CLASS_LABELS)
    gen_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    disc_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

    # fixed noise vector and labels for visualization during training
    fixed_z = tf.random.normal([16, LATENT_DIM])
    fixed_label = tf.constant([i % CLASS_LABELS for i in range(16)], dtype=tf.int32)

    gen_losses, disc_losses, gen_grad_norms, disc_grad_norms = train_model(
        model, gen_optimizer, disc_optimizer, train_dataset, NUM_EPOCHS, fixed_z, fixed_label,
        loss_type=LOSS_TYPE, lambda_gp=10.0, clip_norm=True, max_norm=1.0
    )

    # After training, save the model
    model.gen_model.save("cgan_generator.h5")
    model.disc_model.save("cgan_discriminator.h5")
    # Save full model weights (includes embeddings)
    model.save_weights('cgan_full_weights.ckpt')

    plot_generated_images_tf(model, digit_label=1, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1, save_path='gen_digit_1.png')
    plot_generated_images_tf(model, digit_label=3, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1, save_path='gen_digit_3.png')
    plot_generated_images_tf(model, digit_label=6, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1, save_path='gen_digit_6.png')
    plot_generated_images_tf(model, digit_label=7, latent_dim=LATENT_DIM, grid_dim=15, dim1=0, dim2=1, save_path='gen_digit_7.png')

