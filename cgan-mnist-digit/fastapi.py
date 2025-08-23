from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, conint

import os, uuid  # use uuid.uuid4().hex to generate random text at every call

from model import CGAN_MNIST_Digits_2
from utils import plot_generated_images_tf
from demo_script import LATENT_DIM,IMAGE_SHAPE,CLASS_LABELS

model_cgan_trained = CGAN_MNIST_Digits_2(latent_dim=LATENT_DIM, image_size=IMAGE_SHAPE,
                                         dropout=0.45,alpha=0.2, class_labels=CLASS_LABELS)

# replace the internal sub-models with the trained ones
model_cgan_trained.gen_model = keras.models.load_model("generator.keras")
model_cgan_trained.disc_model = keras.models.load_model("discriminator.keras")

# Fast API
app = FastAPI(title="CGAN MNIST Digit Generator")

Digit = conint(ge=0,le=9) # Digits must be between 0 and 9
class InputRequest(BaseModel):
    value: Digit         # Flag input not with[0,9]
    grid : bool = False  # new flag (default False = single image)

@app.post("/gen_images")
def gen_images(req:InputRequest):
    
    os.makedirs("outputs", exist_ok=True)
    filename = f"generated_{req.value}_{uuid.uuid4().hex}.png"
    save_path = os.path.join("outputs", filename)
    
    if req.grid:
        plot_generated_images_tf(model_cgan_trained,digit_label=req.value,
                                 latent_dim=LATENT_DIM, grid_dim=10, dim1=0, dim2=1,save_path=save_path)
    else:  
        # --- generate a single digit ---
        noise = tf.random.normal([1, LATENT_DIM])
        label = tf.constant([req.value], dtype=tf.int32)
        img = model_cgan_trained.generate(noise, label)[0, :, :, 0].numpy()
        img = (img + 1) / 2.0  # scale back to [0,1]
        plt.imsave(save_path, img, cmap="gray")
         
    return FileResponse(save_path, media_type="image/png")
