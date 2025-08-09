import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================
#     Basic WGAN-GP Model
# ============================

class WGAN_GP_Div2k(nn.Module):
    """
    WGAN-GP model for super-resolution without skip connections.
    Contains a generator (gen_model) and discriminator (disc_model).
    """
    def __init__(self, image_channel=3, n_filters=32, dropout=0.2, negative_slope=0.2, **kwargs):
        super().__init__(**kwargs)

        # Generator upscales from (32x32) → (256x256)
        self.gen_model = nn.Sequential(
            nn.ConvTranspose2d(image_channel, n_filters * 4, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 3, 1, 1, bias=False),   # 64x64
            nn.LeakyReLU(negative_slope),

            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),   # 128x128
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(n_filters * 2, n_filters * 2, 3, 1, 1, bias=False),   # 128x128
            nn.LeakyReLU(negative_slope),

            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),       # 256x256
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(n_filters, n_filters, 3, 1, 1, bias=False),           # 256x256
            nn.LeakyReLU(negative_slope),

            nn.ConvTranspose2d(n_filters, image_channel, 3, 1, 1, bias=False),       # output
            nn.Tanh()
        )

        # Discriminator downsamples from 256x256 to 8x8
        self.disc_model = nn.Sequential(
            nn.Conv2d(image_channel, n_filters // 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters // 2), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters // 2, n_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters, n_filters * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 2), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 4), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters * 4, image_channel, 3, 2, 1, bias=False),
        )

    def generate(self, z):
        return self.gen_model(z)

    def discriminate(self, x):
        return self.disc_model(x)

    def generate_images(self, test_sample):
        """
        Displays a grid of generated images.
        """
        pred = self.generate(test_sample.to(device))
        plt.figure(figsize=(4, 4))
        for i, data in enumerate(pred):
            plt.subplot(4, 4, i + 1)
            plt.imshow(data.detach().cpu().permute(1, 2, 0).numpy())
            plt.axis('off')
        plt.show()


# ============================
#   Generator with Skip #1
# ============================

class GeneratorWithSkip_1(nn.Module):
    """
    Generator that includes optional skip connections between input and upsampling stages.
    """
    def __init__(self, image_channel=3, n_filters=32, negative_slope=0.2, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(image_channel, n_filters * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(n_filters * 2, n_filters * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ConvTranspose2d(n_filters, n_filters, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True)
        )

        in_final = n_filters if not skip_connect else n_filters + image_channel
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_final, image_channel, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        input_x = x
        out1 = self.up1(x)
        out2 = self.up2(out1)
        out3 = self.up3(out2)

        if self.skip_connect:
            input_up = F.interpolate(input_x, size=out3.shape[-2:], mode='bilinear', align_corners=False)
            out3 = torch.cat([out3, input_up], dim=1)

        return self.final(out3)


# ============================
#   Generator with Skip #2
# ============================

class GeneratorWithSkip_2(nn.Module):
    """
    An enhanced generator model with skip connections at multiple resolution stages.
    """
    def __init__(self, image_channel=3, n_filters=32, negative_slope=0.2, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect

        self.up1_conv1 = nn.ConvTranspose2d(image_channel, n_filters * 4, 4, 2, 1, bias=False)
        self.up1_relu1 = nn.LeakyReLU(negative_slope, inplace=True)
        self.up1_conv2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 3, 1, 1, bias=False)
        self.up1_relu2 = nn.LeakyReLU(negative_slope, inplace=True)

        self.up2_conv1 = nn.ConvTranspose2d(n_filters * 4 + image_channel if skip_connect else n_filters * 4,
                                            n_filters * 2, 4, 2, 1, bias=False)
        self.up2_relu1 = nn.LeakyReLU(negative_slope, inplace=True)
        self.up2_conv2 = nn.ConvTranspose2d(n_filters * 2, n_filters * 2, 3, 1, 1, bias=False)
        self.up2_relu2 = nn.LeakyReLU(negative_slope, inplace=True)

        self.up3_conv1 = nn.ConvTranspose2d(n_filters * 2 + image_channel if skip_connect else n_filters * 2,
                                            n_filters, 4, 2, 1, bias=False)
        self.up3_relu1 = nn.LeakyReLU(negative_slope, inplace=True)
        self.up3_conv2 = nn.ConvTranspose2d(n_filters, n_filters, 3, 1, 1, bias=False)
        self.up3_relu2 = nn.LeakyReLU(negative_slope, inplace=True)

        self.final_conv = nn.ConvTranspose2d(n_filters + image_channel if skip_connect else n_filters,
                                             image_channel, 3, 1, 1, bias=False)
        self.final_act = nn.Tanh()

    def forward(self, x):
        input_x = x

        out1 = self.up1_conv1(x)
        out1 = self.up1_relu1(out1)
        out1 = self.up1_conv2(out1)
        out1 = self.up1_relu2(out1)

        if self.skip_connect:
            x_up1 = F.interpolate(input_x, size=out1.shape[-2:], mode='bilinear', align_corners=False)
            out1 = torch.cat([out1, x_up1], dim=1)

        out2 = self.up2_conv1(out1)
        out2 = self.up2_relu1(out2)
        out2 = self.up2_conv2(out2)
        out2 = self.up2_relu2(out2)

        if self.skip_connect:
            x_up2 = F.interpolate(input_x, size=out2.shape[-2:], mode='bilinear', align_corners=False)
            out2 = torch.cat([out2, x_up2], dim=1)

        out3 = self.up3_conv1(out2)
        out3 = self.up3_relu1(out3)
        out3 = self.up3_conv2(out3)
        out3 = self.up3_relu2(out3)

        if self.skip_connect:
            x_up3 = F.interpolate(input_x, size=out3.shape[-2:], mode='bilinear', align_corners=False)
            out3 = torch.cat([out3, x_up3], dim=1)

        out = self.final_conv(out3)
        out = self.final_act(out)
        return out


# ============================
#    WGAN-GP with Skip #2
# ============================

class WGAN_GP_Div2k_2(nn.Module):
    """
    WGAN-GP wrapper using GeneratorWithSkip_2.
    """
    def __init__(self, image_channel=3, n_filters=32, dropout=0.2, negative_slope=0.2, skip_connect=True):
        super().__init__()

        self.gen_model = GeneratorWithSkip_2(image_channel, n_filters, negative_slope, skip_connect)

        self.disc_model = nn.Sequential(
            nn.Conv2d(image_channel, n_filters // 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters // 2), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters // 2, n_filters, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters, n_filters * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 2), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 4), nn.LeakyReLU(negative_slope), nn.Dropout(dropout),

            nn.Conv2d(n_filters * 4, image_channel, 3, 2, 1, bias=False),
        )

    def generate(self, z):
        return self.gen_model(z)

    def discriminate(self, x):
        return self.disc_model(x)

    def generate_images(self, test_sample):
        """
        Displays a grid of generated images.
        """
        pred = self.generate(test_sample.to(device))
        plt.figure(figsize=(4, 4))
        for i, data in enumerate(pred):
            plt.subplot(4, 4, i + 1)
            plt.imshow(data.detach().cpu().permute(1, 2, 0).numpy())
            plt.axis('off')
        plt.show()
