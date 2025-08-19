"""
model.py

Defines the CycleGAN generator and discriminator models,
along with helper layers such as InstanceNormalization,
downsampling, and upsampling blocks.
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# -----------------------------------
# Custom Normalization Layer
# -----------------------------------

class InstanceNormalization(keras.layers.Layer):
    """
    Custom Instance Normalization Layer.
    """
    def __init__(self, epsi=1e-9, **kwargs):
        super().__init__(**kwargs)
        self.epsi = epsi

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            trainable=True,
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02)
        )
        self.shift = self.add_weight(
            name='shift',
            trainable=True,
            shape=input_shape[-1:],
            initializer='zeros'
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        stddev = tf.math.rsqrt(self.epsi + variance)
        x = (x - mean) * stddev
        return self.scale * x + self.shift

# -----------------------------------
# Downsample Block
# -----------------------------------

def downsample(filters, kernel_size, norm_type='batchnorm', apply_norm=True):
    """
    Creates a downsampling block using Conv2D.
    """
    assert norm_type in ["batchnorm", "instancenorm"], "Invalid norm_type"
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = keras.Sequential()
    result.add(keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=2,
        padding='same',
        kernel_initializer=initializer
    ))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(keras.layers.LeakyReLU())
    return result

# -----------------------------------
# Upsample Block
# -----------------------------------

def upsample(filters, kernel_size, norm_type='batchnorm', apply_dropout=False):
    """
    Creates an upsampling block using Conv2DTranspose.
    """
    assert norm_type in ["batchnorm", "instancenorm"], "Invalid norm_type"
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=2,
        padding='same',
        kernel_initializer=initializer
    ))

    if norm_type.lower() == 'batchnorm':
        result.add(keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))

    result.add(keras.layers.ReLU())
    return result

# -----------------------------------
# CycleGAN Generator
# -----------------------------------

class CycleGAN_Generator(keras.Model):
    """
    U-Net style Generator model for CycleGAN.
    """
    def __init__(self, input_image_size=(256, 256, 3), norm_type='instancenorm', **kwargs):
        super().__init__(**kwargs)
        self.input_image_size = input_image_size
        self.norm_type = norm_type

        # Encoder (downsampling)
        self.enc_gen = keras.Sequential([
            downsample(64, 4, apply_norm=False),
            downsample(128, 4, norm_type=norm_type),
            downsample(256, 4, norm_type=norm_type),
            downsample(512, 4, norm_type=norm_type),
            downsample(512, 4, norm_type=norm_type),
            downsample(512, 4, norm_type=norm_type),
            downsample(512, 4, norm_type=norm_type),
            downsample(512, 4, norm_type=norm_type),
        ])

        # Decoder (upsampling)
        self.dec_gen = keras.Sequential([
            upsample(512, 4, norm_type=norm_type, apply_dropout=True),
            upsample(512, 4, norm_type=norm_type, apply_dropout=True),
            upsample(512, 4, norm_type=norm_type, apply_dropout=True),
            upsample(512, 4, norm_type=norm_type),
            upsample(256, 4, norm_type=norm_type),
            upsample(128, 4, norm_type=norm_type),
            upsample(64, 4, norm_type=norm_type),
        ])

        # Final output layer
        self.gen_last = keras.layers.Conv2DTranspose(
            self.input_image_size[-1],
            kernel_size=4,
            strides=2,
            padding='same',
            activation='tanh'
        )

        self.concat = keras.layers.Concatenate()

    def call(self, inputs):
        x = inputs
        skips = []

        # Forward pass through encoder
        for layer in self.enc_gen.layers:
            x = layer(x)
            skips.append(x)

        skips = reversed(skips[:-1])  # Skip the last layer

        # Forward pass through decoder with skip connections
        for layer, skip in zip(self.dec_gen.layers, skips):
            x = layer(x)
            x = self.concat([x, skip])

        return self.gen_last(x)

    def generate_images(self, test_sample):
        """
        Displays input and generated images side-by-side.
        """
        titles = ['Input', 'Generated']
        pred = self(test_sample)
        images = [test_sample, pred]
        plt.figure(figsize=(15, 15))
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow((images[i][0] + 1) / 2.0)
            plt.axis('off')
            plt.title(titles[i])
        plt.show()

    def plot_training_progress(self, test_sample, target_image):
        """
        Displays input, generated, and target images.
        """
        titles = ['Input', 'Generated', 'Target']
        pred = self(test_sample)
        images = [test_sample, pred, target_image]
        plt.figure(figsize=(15, 15))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow((images[i][0] + 1) / 2.0)
            plt.axis('off')
            plt.title(titles[i])
        plt.show()

# -----------------------------------
# CycleGAN Discriminator
# -----------------------------------

class CycleGAN_Discriminator(keras.Model):
    """
    PatchGAN Discriminator for CycleGAN.
    """
    def __init__(self, norm_type='instancenorm', **kwargs):
        super().__init__(**kwargs)
        initializer = keras.initializers.RandomNormal(stddev=0.02)

        self.model = keras.Sequential([
            downsample(64, 4, apply_norm=False),
            downsample(128, 4, norm_type=norm_type),
            downsample(256, 4, norm_type=norm_type),
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False),
            InstanceNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.ZeroPadding2D(padding=1),
            keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        ])

    def call(self, input_image):
        return self.model(input_image)
