import tensorflow as tf
import pathlib

# Dataset path (Kaggle specific, modify for local use if needed)
PATH = pathlib.Path("/kaggle/input/pix2pix-facades-dataset/facades")

# Image and dataset constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# ------------------------ Image Processing Functions ------------------------

def load(image_file):
    """
    Loads and splits an image file into input (B) and real (A) images.
    """
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split the image in half
    w = tf.shape(image)[1] // 2
    input_image = tf.cast(image[:, w:, :], tf.float32)
    real_image = tf.cast(image[:, :w, :], tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    """
    Resizes input and real images to the given height and width using nearest neighbor.
    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    """
    Applies the same random crop to both input and real images.
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    """
    Normalizes input and real images to range [-1, 1].
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    """
    Applies data augmentation: resize → random crop → random horizontal flip.
    """
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    """
    Loads and processes a training image file with data augmentation.
    """
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    """
    Loads and processes a test image file (no augmentation).
    """
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


# ------------------------ Dataset Creation ------------------------

# Paths to train/test/val folders
train_path = PATH / "train"
test_path = PATH / "test"
val_path = PATH / "val"

# If no training images found, warn the user
if not any(train_path.glob("*.jpg")):
    print("No training images found! Check your PATH:", train_path)
else:
    print(f"Found {len(list(train_path.glob('*.jpg')))} training images.")

# Load and combine train and val datasets
train_files = tf.data.Dataset.list_files(str(train_path / "*.jpg"), shuffle=True)
val_files = tf.data.Dataset.list_files(str(val_path / "*.jpg"), shuffle=True)
combined_train_files = train_files.concatenate(val_files)

# Build training dataset
train_dataset = combined_train_files.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)

# Build test dataset
test_files = tf.data.Dataset.list_files(str(test_path / "*.jpg"))
test_dataset = test_files.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Quick check
for input_image, real_image in train_dataset.take(1):
    print("Input image shape:", input_image.shape)
    print("Real image shape:", real_image.shape)

print(f"Train length: {len(train_dataset)} | Test length: {len(test_dataset)}")
