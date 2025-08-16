# model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class CGAN_MNIST_Fashion(keras.Model):
    """
    Conditional GAN for Fashion-MNIST.
    
    Generates images conditioned on class labels using a U-Net-style generator
    and PatchGAN-style discriminator.
    """
    def __init__(self, latent_dim=32, num_feature_map=128, image_size=(28, 28, 1),
                 class_labels=10, alpha=0.2, momentum=0.8, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.class_labels = class_labels
        out_embed_dim = latent_dim // 2

        # Generator components
        self.embed_gen = keras.Sequential([
            keras.layers.Embedding(input_dim=class_labels, output_dim=out_embed_dim),
            keras.layers.Dense(7*7),
            keras.layers.Reshape((7, 7, 1))
        ])

        self.latent_image = keras.Sequential([
            keras.layers.InputLayer(shape=(latent_dim,)),
            keras.layers.Dense(num_feature_map * 7 * 7),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Reshape((7, 7, num_feature_map))
        ])

        self.gen_model = keras.Sequential([
            keras.layers.Conv2DTranspose(num_feature_map, kernel_size=4, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Conv2DTranspose(num_feature_map, kernel_size=4, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='tanh')
        ])

        # Discriminator components
        self.embed_disc = keras.Sequential([
            keras.layers.Embedding(input_dim=class_labels, output_dim=out_embed_dim),
            keras.layers.Dense(np.prod(image_size)),
            keras.layers.Reshape(image_size)
        ])

        self.disc_model = keras.Sequential([
            keras.layers.InputLayer(shape=(28, 28, 2)),
            keras.layers.Conv2D(num_feature_map, kernel_size=3, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Conv2D(num_feature_map, kernel_size=3, strides=2, padding='same'),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def generate(self, z, label):
        """
        Generate fake image from latent vector and label.
        """
        cond = self.embed_gen(label)
        z = self.latent_image(z)
        z_cond = keras.layers.Concatenate()([z, cond])
        return self.gen_model(z_cond)

    def discriminate(self, image, label):
        """
        Discriminate real/fake image based on label.
        """
        cond = self.embed_disc(label)
        image_cond = keras.layers.Concatenate()([image, cond])
        return self.disc_model(image_cond)

    def generate_images(self, test_sample, rand_label):
        """
        Utility for displaying generated samples in a 4x4 grid.
        """
        pred = self.generate(test_sample, rand_label)
        fig = plt.figure(figsize=(4, 4))
        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((pred[i, :, :, 0] + 1) / 2.0, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
