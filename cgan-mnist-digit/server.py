# server.py
# server.py
import io
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, conint

from model import CGAN_MNIST_Digits_2   # your class
from utils import plot_generated_images_tf

# --- constants ---
LATENT_DIM   = 100
IMAGE_SHAPE  = (28, 28, 1)
CLASS_LABELS = 10

app = FastAPI(title="CGAN MNIST Digit Generator")

Digit = conint(ge=0, le=9)
class InputRequest(BaseModel):
    value: Digit
    grid: bool = False

# Load trained model parts
model_cgan_trained = CGAN_MNIST_Digits_2(
    latent_dim=LATENT_DIM, image_size=IMAGE_SHAPE,
    dropout=0.45, alpha=0.2, class_labels=CLASS_LABELS
)

model_cgan_trained.embed_gen    = keras.models.load_model("embed_gen.keras")
model_cgan_trained.latent_image = keras.models.load_model("latent_image.keras")
model_cgan_trained.gen_model    = keras.models.load_model("generator.keras")
model_cgan_trained.embed_disc   = keras.models.load_model("embed_disc.keras")
model_cgan_trained.disc_model   = keras.models.load_model("discriminator.keras")


@app.post("/gen_images")
def gen_images(req: InputRequest):
    buf = io.BytesIO()

    if req.grid:
        # Generate a 10x10 grid
        plot_generated_images_tf(
            model_cgan_trained,
            digit_label=int(req.value),
            latent_dim=LATENT_DIM,
            grid_dim=10,
            dim1=0, dim2=1,
            save_path=buf
        )
    else:
        # Single digit
        noise = tf.random.normal([1, LATENT_DIM])
        label = tf.constant([int(req.value)], dtype=tf.int32)
        img = model_cgan_trained.generate(noise, label)[0, :, :, 0].numpy()
        img = (img + 1) / 2.0
        plt.imsave(buf, img, cmap="gray", format="png")

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
