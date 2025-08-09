import time
import random
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import grad as torch_grad
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_super_resolution(model, dataloader=None, low_res=None, high_res=None, 
                           num_images=8, calculate_quantitative=False, random_sample=False):
    model.eval()
    with torch.no_grad():
        if dataloader is not None:
            if random_sample:
                low_res, high_res = random.choice(list(dataloader))
            else:
                low_res, high_res = next(iter(dataloader))
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        
        if calculate_quantitative:
            generated = model.generate(low_res)
            psnr_total, ssim_total = 0.0, 0.0

            gen_clamped = (generated * 0.5 + 0.5).clamp(0, 1)   # Convert to [0,1] before evaluation
            high_clamped = (high_res * 0.5 + 0.5).clamp(0, 1)
            num_images_2 = len(gen_clamped)
            for gen_img, real_img in zip(gen_clamped, high_clamped):
                gen_np = gen_img.permute(1, 2, 0).cpu().numpy()
                real_np = real_img.permute(1, 2, 0).cpu().numpy()

                psnr_val = peak_signal_noise_ratio(real_np, gen_np, data_range=1.0)
                ssim_val = ssim(real_np, gen_np, data_range=1.0, win_size=5, channel_axis=-1)

                psnr_total += psnr_val
                ssim_total += ssim_val

            psnr_value = psnr_total / num_images_2
            ssim_value = ssim_total / num_images_2

            generated = generated[:num_images]
            low_res = low_res[:num_images]
            high_res = high_res[:num_images]
            img_tit = "Top: Low-Res Upsampled |  Middle: Generated |  Bottom: High-Res Ground Truth\n"\
                      f"Validation dataset: PSNR {psnr_value:.4f}  |  SSIM {ssim_value:.4f}"
        else:
            high_res = high_res[:num_images]
            low_res = low_res[:num_images]
            generated = model.generate(low_res)  # Generate high-res from low-res
            img_tit = "Top: Low-Res Upsampled | Middle: Generated | Bottom: High-Res Ground Truth"
        
        # Upsample low-res to match high-res size for better comparison
        low_res_upsampled = F.interpolate(low_res, size=high_res.shape[-2:], mode='bilinear', align_corners=False)

        # Clamp/Tanh normalization handling
        low_res_upsampled = (low_res_upsampled * 0.5 + 0.5).clamp(0, 1)
        high_res = (high_res * 0.5 + 0.5).clamp(0, 1)
        generated = (generated * 0.5 + 0.5).clamp(0, 1)

        # Create a grid: Top - low_res_upsampled | Middle - generated | Bottom - ground truth
        images = torch.cat([low_res_upsampled, generated, high_res], dim=0)
        grid = torchvision.utils.make_grid(images.cpu(), nrow=num_images, padding=2)

        plt.figure(figsize=(20, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(img_tit, fontsize=14)
        plt.show()
        
def psnr(img1, img2):
    """Measures how much difference there is between the generated image and 
       the ground truth image. Higher PSNR(Peak-Signal-to-Noise-Ratio) means better quality
    """
    mse = F.mse_loss(img1, img2)
    psnr_value = 10 * torch.log10(1.0 / mse)  # Assuming images are in [0, 1] range
    return psnr_value

def calculate_ssim(img1, img2):
    """Measures the perceptual similarity between two images, taking luminance, contrast, and structure into account. 
        A higher SSIM(Structural Similarity Index Measure) means a closer match to the ground truth.
    """
    return ssim(img1, img2, multichannel=True)
    
def penalty_gradients(model, real_image, generated_image, lambda_gp):
    batch_size, image_channel, height, width = real_image.shape
    real_image = real_image.to(device)
    generated_image = generated_image.to(device)
   
    alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
    interpolated = (alpha * real_image + (1 - alpha) * generated_image).requires_grad_(True)
    interpolated = interpolated.to(device)
    
    proba_interpolated = model.discriminate(interpolated)

    gradients = torch_grad(outputs=proba_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones_like(proba_interpolated).to(device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients - 1) ** 2).mean()

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
    
def train_wgan_with_gp_model(model, gen_optimizer, disc_optimizer, num_epochs, train_dl, valid_dl, lambda_gp=10,
                             clip_norm_disc=False, clip_norm_gen=False, norm_max_disc=5, norm_max_gen=5):
    epoch_samples = []
    gen_losses_train, disc_losses_train = [], []  
    gen_losses_valid, disc_losses_valid = [], []  
    start = time.time()
    for epoch in range(num_epochs):
        g_loss_t, d_loss_t, g_loss_v, d_loss_v = 0.0, 0.0, 0.0, 0.0
        n = 0
        for low_res, high_res in train_dl:  
            low_res = low_res.to(device)      #  [B, 3, 8, 8]
            high_res = high_res.to(device)    #  [B,3,256,256]

            # === Train Discriminator ===
            disc_optimizer.zero_grad()
            
            real_preds = model.discriminate(high_res)       # Discriminator loss on real images
            fake_data = model.generate(low_res)             # generate fake image  
            fake_preds = model.discriminate(fake_data)      # Discriminator loss on fake images 
            gradient_penalty = penalty_gradients(model, high_res, fake_data, lambda_gp)
            fake_loss = fake_preds.mean()
            real_loss = -real_preds.mean()
            disc_loss = real_loss + fake_loss + gradient_penalty
            disc_loss.backward()
            if clip_norm_disc:
                torch.nn.utils.clip_grad_norm_(model.disc_model.parameters(), max_norm=norm_max_disc)
            disc_grad_norm = grad_norm(model.disc_model)
            disc_optimizer.step()
            d_loss_t += disc_loss
        
            # === Train Generator ===
            gen_optimizer.zero_grad()
            
            fake_data = model.generate(low_res)  
            fake_preds = model.discriminate(fake_data) 
            gen_loss = -fake_preds.mean()
            gen_loss.backward()
            gen_grad_norm = grad_norm(model.gen_model)
            if clip_norm_gen:
                torch.nn.utils.clip_grad_norm_(model.gen_model.parameters(), max_norm=norm_max_gen)
            gen_optimizer.step()
            g_loss_t += gen_loss

            if n % 100 == 0:
                print('.', end='')
            n += 1
        d_loss_t /= len(train_dl)
        g_loss_t /= len(train_dl)
        gen_losses_train.append(g_loss_t.item())
        disc_losses_train.append(d_loss_t.item())

        
        with torch.no_grad():
            for low_res, high_res in valid_dl:  
                low_res = low_res.to(device)      #  [B, 3, 8, 8]
                high_res = high_res.to(device)    #  [B,3,256,256]
    
                # === Train Discriminator ===
                real_preds = model.discriminate(high_res)       # Discriminator loss on real images
                fake_data = model.generate(low_res)             # generate fake image  
                fake_preds = model.discriminate(fake_data)      # Discriminator loss on fake images 
                fake_loss = fake_preds.mean()
                real_loss = -real_preds.mean()
                disc_loss = real_loss + fake_loss 
                d_loss_v += disc_loss
    
                # === Train Generator ===
                fake_data = model.generate(low_res)  
                fake_preds = model.discriminate(fake_data) 
                gen_loss = -fake_preds.mean()
                g_loss_v += gen_loss
            d_loss_v /= len(valid_dl)
            g_loss_v /= len(valid_dl)
            gen_losses_valid.append(g_loss_v.item())
            disc_losses_valid.append(d_loss_v.item())
            
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            elapsed = (time.time() - start)/60
            print(f"Epoch [{epoch+1}/{num_epochs}]\n"
                  f"  Train - Disc: {d_loss_t.item():.4f}, Gen: {g_loss_t.item():.4f}\n"
                  f"  Valid - Disc: {d_loss_v.item():.4f}, Gen: {g_loss_v.item():.4f}\n"
                  f"  Grad Norms - Disc: {disc_grad_norm:.4f}, Gen: {gen_grad_norm:.4f} | Time: {elapsed:.2f} min")

            show_super_resolution(model, dataloader=valid_dl, num_images=8, calculate_quantitative=True)
            start = time.time()

    return gen_losses_train, disc_losses_train, gen_losses_valid, disc_losses_valid

##############################################################################################################
def train_wgan_with_gp_model_2(model, gen_optimizer, disc_optimizer, num_epochs, train_dl, valid_dl, lambda_gp=10,
                             clip_norm_disc=False, clip_norm_gen=False, norm_max_disc=5.0, norm_max_gen=5.0,
                             n_critic=2, apply_recon_loss=True, recon_loss_weight=100.0, apply_noise_to_real=True):
    epoch_samples = []
    gen_losses_train, disc_losses_train = [], []
    gen_losses_valid, disc_losses_valid = [], []
    start = time.time()

    for epoch in range(num_epochs):
        g_loss_t, d_loss_t, g_loss_v, d_loss_v = 0.0, 0.0, 0.0, 0.0
        disc_grad_norm = gen_grad_norm = 0.0
        
        for i, (low_res, high_res) in enumerate(train_dl):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # === Train Discriminator ===
            if i % n_critic == 0:
                disc_optimizer.zero_grad()

                # Optional noise to real images
                if apply_noise_to_real:
                    real_input = high_res + 0.05 * torch.randn_like(high_res)
                else:
                    real_input = high_res

                real_preds = model.discriminate(real_input)
                fake_data = model.generate(low_res).detach()
                fake_preds = model.discriminate(fake_data)

                gradient_penalty = penalty_gradients(model, real_input, fake_data, lambda_gp)

                fake_loss = fake_preds.mean()
                real_loss = -real_preds.mean()
                disc_loss = real_loss + fake_loss + gradient_penalty

                disc_loss.backward()
                if clip_norm_disc:
                    torch.nn.utils.clip_grad_norm_(model.disc_model.parameters(), max_norm=norm_max_disc)
                disc_grad_norm = grad_norm(model.disc_model)
                disc_optimizer.step()
                d_loss_t += disc_loss

            # === Train Generator ===
            gen_optimizer.zero_grad()

            fake_data = model.generate(low_res)
            fake_preds = model.discriminate(fake_data)
            gen_loss = -fake_preds.mean()

            if apply_recon_loss:
                recon_loss = F.l1_loss(fake_data, high_res)
                gen_loss += recon_loss_weight * recon_loss

            gen_loss.backward()
            if clip_norm_gen:
                torch.nn.utils.clip_grad_norm_(model.gen_model.parameters(), max_norm=norm_max_gen)
            gen_grad_norm = grad_norm(model.gen_model)
            gen_optimizer.step()
            g_loss_t += gen_loss

            if i % 100 == 0:
                print('.', end='')

        d_loss_t /= len(train_dl)
        g_loss_t /= len(train_dl)
        gen_losses_train.append(g_loss_t.item())
        disc_losses_train.append(d_loss_t.item())

        # === Validation ===
        with torch.no_grad():
            for low_res, high_res in valid_dl:
                low_res = low_res.to(device)
                high_res = high_res.to(device)

                real_preds = model.discriminate(high_res)
                fake_data = model.generate(low_res)
                fake_preds = model.discriminate(fake_data)

                fake_loss = fake_preds.mean()
                real_loss = -real_preds.mean()
                disc_loss = real_loss + fake_loss
                d_loss_v += disc_loss

                gen_loss = -fake_preds.mean()
                if apply_recon_loss:
                    recon_loss = F.l1_loss(fake_data, high_res)
                    gen_loss += recon_loss_weight * recon_loss
                g_loss_v += gen_loss

            d_loss_v /= len(valid_dl)
            g_loss_v /= len(valid_dl)
            gen_losses_valid.append(g_loss_v.item())
            disc_losses_valid.append(d_loss_v.item())

        # === Logging ===
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            elapsed = (time.time() - start) / 60
            print(f"\nEpoch [{epoch+1}/{num_epochs}]\n"
                  f"  Train - Disc: {d_loss_t.item():.4f}, Gen: {g_loss_t.item():.4f}\n"
                  f"  Valid - Disc: {d_loss_v.item():.4f}, Gen: {g_loss_v.item():.4f}\n"
                  f"  Grad Norms - Disc: {disc_grad_norm:.4f}, Gen: {gen_grad_norm:.4f} | Time: {elapsed:.2f} min")
            show_super_resolution(model, dataloader=valid_dl, num_images=8, calculate_quantitative=True)
            start = time.time()

    return gen_losses_train, disc_losses_train, gen_losses_valid, disc_losses_valid
