# Generative-Adversarial-Networks(GANs) Projects

  A collection of Generative Adversarial Networks (GANs) projects implemented from scratch
  using Python, TensorFlow, and PyTorch. These projects demonstrate foundational concepts,
  advanced architectures, and practical applications of GANs.
  
---
## Projects:

### 1. Conditional GAN (CGAN)

      Generates realistic, class-conditional images for MNIST or Fashion-MNIST datasets.
      Demonstrates controlled image generation using vanilla GAN loss.
      
    features:
      - Class-conditional image generation
      - Vanilla GAN loss
      - Sharp image generation in 1–3 epochs
      - Deployed on Render for live testing
      
---

### 2. CycleGAN

      Performs unpaired image-to-image translation (e.g., horses ↔ zebras).
      Demonstrates unsupervised image translation with cycle-consistency.
      
    features:
      - Unpaired image translation
      - Cycle-consistency loss
      - TensorFlow implementation
---

 ### 3. Pix2Pix
    
      Paired image-to-image translation using CMP Facade dataset (photos ↔ semantic labels).
      Demonstrates supervised conditional GAN for structured image translation tasks.
      
    features:
      - U-Net generator
      - PatchGAN discriminator
      - TensorFlow 2.x implementation

---

 ### 4. Wasserstein GAN with Gradient Penalty (WGAN-GP)
    
      Single-image super-resolution from low to high resolution.
      Demonstrates stable GAN training with gradient penalty.
      
    features:
      - High-quality image generation
      - Gradient penalty for training stability
      - Applications: medical imaging, satellite imagery
      
---

 ### 5. Comparative Study: MLP-GAN vs DCGAN
    
      Compares fully connected GAN (MLP-GAN) vs Deep Convolutional GAN (DCGAN) on MNIST.
      Demonstrates the effect of architecture choice on generated image quality.
      
    features:
      - MLP-GAN: fully connected
      - DCGAN: convolutional
      - Training progression visualization

---

## General Features:

  - From-scratch implementation using TensorFlow / PyTorch
  - Modular, well-documented, suitable for learning or research
  - Applications in image generation, super-resolution, and style transfer
  - Training visualizations, loss curves, and sample outputs
---

## References:
  - GANs in TensorFlow Tutorial: https://www.tensorflow.org/tutorials/generative/dcgan
  - Pix2Pix Paper: https://arxiv.org/abs/1611.07004
  - CycleGAN Paper: https://arxiv.org/abs/1703.10593
  - Wasserstein GAN Paper: https://arxiv.org/abs/1704.00028
