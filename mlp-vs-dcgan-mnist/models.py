import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure the model runs on the correct device (add this if it's not already declared elsewhere)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN_NN_MNIST(nn.Module):
    """
    A simple MLP-based GAN for generating MNIST-like images.
    Contains a generator and discriminator, both using fully connected layers.
    """

    def __init__(self, input_size, gen_hidden_units, gen_num_layers, gen_output_size, 
                 disc_hidden_units, disc_num_layers, disc_output_size=2, dropout=0.1, **kwargs):
        """
        Initialize the MLP GAN.

        Args:
            input_size (int): Size of the noise vector for the generator.
            gen_hidden_units (int): Hidden layer size for generator.
            gen_num_layers (int): Number of hidden layers in generator.
            gen_output_size (int): Output dimension of generator (usually 784 for 28x28 images).
            disc_hidden_units (int): Hidden layer size for discriminator.
            disc_num_layers (int): Number of hidden layers in discriminator.
            disc_output_size (int): Final output size of discriminator (usually 2 for real/fake).
            dropout (float): Dropout rate for discriminator.
        """
        super().__init__(**kwargs)
        
        self.gen_model = nn.Sequential()   # Generator model
        gen_in = input_size
        for i in range(gen_num_layers):
            self.gen_model.add_module(f'fc_g{i}', nn.Linear(gen_in, gen_hidden_units))
            # self.gen_model.add_module(f'batch_norm{i}', nn.BatchNorm1d(gen_hidden_units))  # Optional batch norm
            self.gen_model.add_module(f'lrelu_g{i}', nn.LeakyReLU())
            gen_in = gen_hidden_units
        self.gen_model.add_module(f'fc_g{gen_num_layers}', nn.Linear(gen_in, gen_output_size))
        self.gen_model.add_module('tanh_g', nn.Tanh())
        
        self.disc_model = nn.Sequential()  # Discriminator model
        disc_in = gen_output_size
        for i in range(disc_num_layers):
            self.disc_model.add_module(f'fc_d{i}', nn.Linear(disc_in, disc_hidden_units))
            # self.disc_model.add_module(f'batch_norm{i}', nn.BatchNorm1d(disc_hidden_units))  # Optional batch norm
            self.disc_model.add_module(f'relu_d{i}', nn.LeakyReLU())
            self.disc_model.add_module(f'dropout_d{i}', nn.Dropout(dropout))
            disc_in = disc_hidden_units
        self.disc_model.add_module(f'fc_d{disc_num_layers}', nn.Linear(disc_in, disc_output_size))

    def generate(self, z):
        """
        Generate fake images from latent noise vector.
        
        Args:
            z (Tensor): Latent vector of shape (batch_size, input_size)
        
        Returns:
            Tensor: Generated fake images
        """
        return self.gen_model(z)

    def discriminate(self, x):
        """
        Discriminate real/fake input data.
        
        Args:
            x (Tensor): Input tensor (real or generated image)
        
        Returns:
            Tensor: Discriminator output
        """
        return self.disc_model(x)

    def generate_images(self, test_sample):
        """
        Generate and visualize sample images from the generator.
        
        Args:
            test_sample (Tensor): A batch of latent vectors
        """
        pred = self.generate(test_sample.to(device))
        plt.figure(figsize=(4, 4))
        for i, data in enumerate(pred):
            plt.subplot(4, 4, i + 1)
            plt.imshow((data.detach().cpu().numpy().reshape(28, 28) + 1) / 2, cmap='gray_r')
            plt.axis('off')
        plt.show()


class DCGAN_MNIST(nn.Module):
    """
    A Deep Convolutional GAN (DCGAN) for generating MNIST images using Conv2D and ConvTranspose2D.
    """

    def __init__(self, input_size, n_filters, dropout=0.2, negative_slope=0.2, **kwargs):
        """
        Initialize the DCGAN.

        Args:
            input_size (int): Latent noise vector depth.
            n_filters (int): Base number of filters for conv layers.
            dropout (float): Dropout rate for discriminator.
            negative_slope (float): Slope for LeakyReLU activations.
        """
        super().__init__(**kwargs)

        # Generator model using ConvTranspose2D layers
        self.gen_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=n_filters * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),  # Output: (n_filters*4, 4, 4)
            nn.BatchNorm2d(num_features=n_filters * 4),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.ConvTranspose2d(in_channels=n_filters * 4, out_channels=n_filters * 2,
                               kernel_size=3, stride=2, padding=1, bias=False),  # Output: (n_filters*2, 7, 7)
            nn.BatchNorm2d(num_features=n_filters * 2),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.ConvTranspose2d(in_channels=n_filters * 2, out_channels=n_filters,
                               kernel_size=4, stride=2, padding=1, bias=False),  # Output: (n_filters, 14, 14)
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.ConvTranspose2d(in_channels=n_filters, out_channels=1,
                               kernel_size=4, stride=2, padding=1, bias=False),  # Output: (1, 28, 28)
            nn.Tanh()
        )

        # Discriminator model using Conv2D layers
        self.disc_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=4, stride=2, padding=1, bias=False),  # Output: (n_filters, 14, 14)
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=n_filters, out_channels=n_filters * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),  # Output: (n_filters*2, 7, 7)
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters * 4,
                      kernel_size=3, stride=2, padding=1, bias=False),  # Output: (n_filters*4, 4, 4)
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=n_filters * 4, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),  # Output: (1, 1, 1)
            nn.Flatten()
        )

    def generate(self, z):
        """
        Generate fake images from latent noise.

        Args:
            z (Tensor): Latent noise vector of shape (batch_size, latent_dim, 1, 1)

        Returns:
            Tensor: Generated image tensor
        """
        return self.gen_model(z)

    def discriminate(self, x):
        """
        Discriminate input image as real or fake.

        Args:
            x (Tensor): Input image tensor

        Returns:
            Tensor: Discriminator logits
        """
        return self.disc_model(x)

    def generate_images(self, test_sample):
        """
        Generate and display images from test latent vectors.

        Args:
            test_sample (Tensor): Latent vectors
        """
        pred = self.generate(test_sample.to(device))
        plt.figure(figsize=(4, 4))
        for i, data in enumerate(pred):
            plt.subplot(4, 4, i + 1)
            plt.imshow((data.detach().cpu().numpy().reshape(28, 28) + 1) / 2, cmap='gray_r')
            plt.axis('off')
        plt.show()
