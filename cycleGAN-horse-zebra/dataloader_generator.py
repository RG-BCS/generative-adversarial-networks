"""
dataloader_generator.py

Handles data loading and preprocessing for the CycleGAN project using the
horse2zebra dataset from TensorFlow Datasets.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Set random seed for reproducibility
tf.random.set_seed(55)

# Configuration constants
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.AUTOTUNE

def load_datasets():
    """
    Loads the horse2zebra dataset and returns train/test splits.
    
    Returns:
        train_horses, train_zebras, test_horses, test_zebras: Preprocessed datasets.
    """
    dataset, metadata = tfds.load(
        'cycle_gan/horse2zebra', 
        with_info=True,
        as_supervised=True
    )
    
    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    # Apply preprocessing
    train_horses = train_horses.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_zebras = train_zebras.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE) \
        .shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) \
        .cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE) \
        .cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_horses, train_zebras, test_horses, test_zebras

def random_crop(image):
    """
    Applies random crop to an image.
    """
    return tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

def random_jitter(image):
    """
    Applies random resizing, cropping, and flipping to an image.
    """
    # Resize to 286x286
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(image):
    """
    Normalizes image pixel values to [-1, 1].
    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def preprocess_image_train(image, label):
    """
    Applies training data preprocessing (jitter + normalization).
    """
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    """
    Applies test data preprocessing (only normalization).
    """
    image = normalize(image)
    return image
