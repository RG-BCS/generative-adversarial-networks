import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# -------------------- Model Utility Blocks --------------------

def down_sample(filters, kernel_size=4, apply_batchnorm=True):
    """
    Creates a downsampling block using Conv2D, BatchNorm (optional), and LeakyReLU.
    """
    initializer = keras.initializers.RandomNormal(stddev=0.02)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                                  kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    return model


def up_sample(filters, kernel_size=4, apply_dropout=False):
    """
    Creates an upsampling block using Conv2DTranspose, BatchNorm, Dropout (optional), and ReLU.
    """
    initializer = keras.initializers.RandomNormal(stddev=0.02)
    model = keras.Sequential()
    model.add(keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                           kernel_initializer=initializer, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    if apply_dropout:
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.ReLU())
    return model


# -------------------- Pix2Pix Model Definition --------------------

class Pix2Pix_Model(keras.Model):
    """
    Defines the Pix2Pix model with U-Net generator and PatchGAN discriminator.
    """
    def __init__(self, input_image_size=(256, 256, 3), **kwargs):
        super().__init__(**kwargs)
        self.input_image_size = input_image_size
        initializer = keras.initializers.RandomNormal(stddev=0.02)

        # Generator Encoder: series of downsampling layers
        self.enc_gen = keras.Sequential([
            down_sample(64, apply_batchnorm=False),  # (bs, 128, 128, 64)
            down_sample(128),                        # (bs, 64, 64, 128)
            down_sample(256),                        # (bs, 32, 32, 256)
            down_sample(512),                        # (bs, 16, 16, 512)
            down_sample(512),                        # (bs, 8, 8, 512)
            down_sample(512),                        # (bs, 4, 4, 512)
            down_sample(512),                        # (bs, 2, 2, 512)
            down_sample(512),                        # (bs, 1, 1, 512)
        ])

        # Generator Decoder: series of upsampling layers
        self.dec_gen = keras.Sequential([
            up_sample(512, apply_dropout=True),  # (bs, 2, 2, 1024)
            up_sample(512, apply_dropout=True),  # (bs, 4, 4, 1024)
            up_sample(512, apply_dropout=True),  # (bs, 8, 8, 1024)
            up_sample(512),                      # (bs, 16, 16, 1024)
            up_sample(256),                      # (bs, 32, 32, 512)
            up_sample(128),                      # (bs, 64, 64, 256)
            up_sample(64),                       # (bs, 128, 128, 128)
        ])

        # Final generator layer to output image with tanh activation
        self.gen_last = keras.layers.Conv2DTranspose(input_image_size[-1], 4, strides=2, padding='same',
                                                     activation='tanh')  # (bs, 256, 256, 3)

        # Discriminator model: PatchGAN
        self.disc_model = keras.Sequential([
            down_sample(64, 4, False),              # (bs, 128, 128, 64)
            down_sample(128, 4),                    # (bs, 64, 64, 128)
            down_sample(256, 4),                    # (bs, 32, 32, 256)
            keras.layers.ZeroPadding2D(padding=1),  # (bs, 34, 34, 256)
            keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False),  # (bs, 31, 31, 256)
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, use_bias=False),  # (bs, 30, 30, 1)
        ])

        self.concat = keras.layers.Concatenate()

    def generator(self, inputs):
        """
        Runs the input through the generator network with skip connections.
        """
        x = inputs
        skips = []
        for layer in self.enc_gen.layers:
            x = layer(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for layer, skip in zip(self.dec_gen.layers, skips):
            x = layer(x)
            x = self.concat([x, skip])  # skip connections

        return self.gen_last(x)

    def discriminator(self, input_image, target_image):
        """
        Discriminator that evaluates the realness of a pair (input, target/generated).
        """
        x = tf.concat([input_image, target_image], axis=-1)
        return self.disc_model(x)

    def generate_images(self, test_sample, target_image):
        """
        Generates and displays input, generated, and target images.
        """
        titles = ['Input', 'Generated', 'Target']
        pred = self.generator(test_sample)
        images = [test_sample, pred, target_image]
        fig = plt.figure(figsize=(15, 15))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow((images[i][0] + 1) / 2.0)  # Rescale from [-1, 1] to [0, 1]
            plt.axis('off')
            plt.title(titles[i])
        plt.show()
