import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image
import torch

# Paths to your datasets
image_path_train = "/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR"
image_path_valid = "/kaggle/input/div2k-high-resolution-images/DIV2K_valid_HR/DIV2K_valid_HR"

# Resolution choices for generator input (e.g., 32x32 low-res images)
gen_input_img_size = (32, 32) 

# Transformations for low-res input images to generator
gen_transform = transforms.Compose([
    transforms.Resize(gen_input_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Transformations for high-res images to discriminator (256x256)
disc_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class Div2k_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size  # Resize all images to target size (high-res size)
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        high_res = PIL.Image.open(img_path).convert('RGB')

        # Resize high-res image to target size (256x256)
        high_res = high_res.resize(self.target_size, PIL.Image.Resampling.LANCZOS)
        # Create low-res version from high-res (32x32)
        low_res = high_res.resize(gen_input_img_size, PIL.Image.Resampling.BICUBIC)

        # Apply transforms
        low_res_image = self.transform(low_res) if self.transform else low_res
        high_res_image = self.target_transform(high_res) if self.target_transform else high_res

        return low_res_image, high_res_image

# Create dataset instances for train and validation
train_ds = Div2k_Dataset(image_path_train, transform=gen_transform, target_transform=disc_transform)
valid_ds = Div2k_Dataset(image_path_valid, transform=gen_transform, target_transform=disc_transform)

# Batch size
BATCH_SIZE = 20

# Create DataLoaders
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
