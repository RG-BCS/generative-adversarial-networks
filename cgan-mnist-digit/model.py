import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


class CGAN_MNIST_Digits_2(keras.Model):
    """
    Conditional GAN model for MNIST digit generation.

    This model learns to generate MNIST digits conditioned on class labels.
    It includes a generator and a discriminator, each using embedding layers
    to incorporate label information directly into the image and latent space.

    Parameters:
        latent_dim (int): Size of the random noise vector.
        image_size (tuple): Shape of the input/output images (default: (28, 28, 1)).
        class_labels (int): Number of distinct labels/classes.
        alpha (float): LeakyReLU negative slope coefficient.
        momentum (float): Momentum for BatchNormalization (if used).
        dropout (float): Dropout rate for the discriminator.
        **kwargs: Additional keyword arguments for keras.Model.
    """

    def __init__(self, latent_dim=32, image_size=(28, 28, 1), class_labels=10,
                 alpha=0.2, momentum=0.8, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.class_labels = class_labels

        # Generator Components

        # Embedding layer for labels: turns class label into a (7x7x1) "image-like" tensor
        self.embed_gen = keras.Sequential([
            keras.layers.Embedding(input_dim=class_labels, output_dim=latent_dim),
            keras.layers.Dense(7 * 7),
            keras.layers.Reshape((7, 7, 1))
        ])

        # Maps latent vector z to initial image feature map (7x7x128)
        self.latent_image = keras.Sequential([
            keras.layers.InputLayer(shape=(latent_dim,)),
            keras.layers.Dense(128 * 7 * 7),
            keras.layers.LeakyReLU(alpha),
            keras.layers.Reshape((7, 7, 128))
        ])

        # Generator upsampling stack
        self.gen_model = keras.Sequential([
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha),  # Output: (14x14)

            keras.layers.Conv2DTranspose(128, kernel_size=1, strides=1, padding='valid'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha),  # Output: (14x14)

            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha),  # Output: (28x28)

            keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='tanh')  # Final output: (28x28x1)
        ])

        # Discriminator Components

        # Embedding label into an image-shaped tensor for concatenation
        self.embed_disc = keras.Sequential([
            keras.layers.Embedding(input_dim=class_labels, output_dim=latent_dim),
            keras.layers.Dense(np.prod(image_size)),
            keras.layers.Reshape(image_size)
        ])

        # Discriminator stack: Conv layers followed by global pooling and dense output
        self.disc_model = keras.Sequential([
            keras.layers.InputLayer(shape=(28, 28, 2)),
            keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha),  # Output: (14x14)

            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha),  # Output: (7x7)

            # keras.layers.Flatten(),  # Removed for better gradient flow
            keras.layers.GlobalMaxPooling2D(),  # More stable alternative to flattening
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1, activation='sigmoid')  # Output probability
        ])

    def generate(self, z, label):
        """
        Generate an image from latent vector z and a class label.

        Args:
            z (Tensor): Latent noise vector (batch_size, latent_dim).
            label (Tensor): Class label (batch_size,).

        Returns:
            Tensor: Generated image (batch_size, 28, 28, 1).
        """
        cond = self.embed_gen(label)
        z = self.latent_image(z)
        z_cond = keras.layers.Concatenate()([z, cond])
        return self.gen_model(z_cond)

    def discriminate(self, image, label):
        """
        Discriminate between real and fake images, conditioned on label.

        Args:
            image (Tensor): Input image (batch_size, 28, 28, 1).
            label (Tensor): Class label (batch_size,).

        Returns:
            Tensor: Probability of image being real (batch_size, 1).
        """
        cond = self.embed_disc(label)
        image_cond = keras.layers.Concatenate()([image, cond])
        return self.disc_model(image_cond)

    def generate_images(self, test_sample, rand_label):
        """
        Visualize generated images for a batch of test samples and labels.

        Args:
            test_sample (Tensor): Latent noise vectors (e.g. shape (16, latent_dim)).
            rand_label (Tensor): Corresponding class labels.

        Displays:
            A grid of generated digit images using matplotlib.
        """
        pred = self.generate(test_sample, rand_label)
        fig = plt.figure(figsize=(4, 4))
        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i + 1)
            # Uncomment below if you want to visualize tanh-scaled output to [0,1]
            # plt.imshow((pred[i, :, :, 0] + 1) / 2.0, cmap='gray')
            plt.imshow(pred[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.show()
