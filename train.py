import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn import metrics

from model import Generator, Encoder, Discriminator
from dataset import get_dataloaders
from losses import gradient_penalty, compute_l1

import matplotlib.pyplot as plt
import torchvision.utils as vutils

def get_anomaly_score(generator, encoder, discriminator, images, lambda_):
    generator.eval()
    encoder.eval()
    discriminator.eval()
    
    with torch.no_grad():
        latent = encoder(images)
        reconstructed_img = generator(latent)
        
        # Features from discriminator (using the return_features flag in your model.py)
        feat_real = discriminator(images, latent, return_features=True)
        feat_recon = discriminator(reconstructed_img, latent, return_features=True)
        
        # Loss R (Pixel) and Loss f_D (Feature)
        l_r = torch.mean(torch.abs(images - reconstructed_img), dim=[1, 2, 3])
        l_fd = torch.mean(torch.abs(feat_real - feat_recon), dim=1)
        
        score = (1 - lambda_) * l_r + lambda_ * l_fd
    return score

def save_reconstruction_grid(real_images, netE, netG, epoch, device, save_path="reconstructions"):
    """
    Creates a grid: Top row = Real Images, Bottom row = Reconstructions.
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    netE.eval()
    netG.eval()
    with torch.no_grad():
        # Get reconstructions: G(E(x))
        recons = netG(netE(real_images[:8])) 
        real_samples = real_images[:8]
        
        # Combine into one grid: Normalize from [-1, 1] to [0, 1] for plotting
        combined = torch.cat([real_samples, recons], dim=0)
        grid = vutils.make_grid(combined, nrow=8, normalize=True, value_range=(-1, 1))
        
        plt.figure(figsize=(12, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Epoch {epoch} Reconstructions (Top: Real, Bottom: G(E(x)))")
        plt.axis('off')
        plt.savefig(f"{save_path}/epoch_{epoch}.png")
        plt.close()

def plot_loss_curves(history, save_path="plots"):
    """
    Plots the training loss trends for Discriminator and Generator/Encoder.
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(history['d_loss'], label='D Loss')
    plt.plot(history['ge_loss'], label='GE Loss')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.title('CBiGAN Training Losses')
    plt.legend()
    plt.savefig(f"{save_path}/loss_curve.png")
    plt.close()



def evaluate(netG, netE, netD, test_loader, device, lambda_, classes):
    all_scores = []
    all_labels = []
    benign_idx = classes.index('benign')

    for images, labels in test_loader:
        images = images.to(device)
        scores = get_anomaly_score(netG, netE, netD, images, lambda_)
        
        # Labels: 0 for benign, 1 for malware (anomaly)
        binary_labels = (labels != benign_idx).int()
        
        all_scores.append(scores.cpu().numpy())
        all_labels.append(binary_labels.numpy())
        
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
    auc = metrics.auc(fpr, tpr)
    print(f">> Eval AUC: {auc:.4f}")



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloaders(args.image_size, args.batch_size)
    
    LATENT_SIZE = 128
    CHANNELS = 3
    IMG_SIZE = 512
    BATCH_SIZE = 4

    # Instantiate models
    netG = Generator(LATENT_SIZE, CHANNELS, upsample_first=False, bn_type='batch')
    netE = Encoder(IMG_SIZE, LATENT_SIZE, bn_type='instance')
    netD = Discriminator(IMG_SIZE, LATENT_SIZE, bn_type='layer')

    optGE = optim.Adam(list(netG.parameters()) + list(netE.parameters()), 
                       lr=args.lr, betas=(float(args.ge_beta1), float(args.ge_beta2) ))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.d_beta1, args.d_beta2))

    history = {'d_loss': [], 'ge_loss': [], 'auc': []}
    for epoch in args.epoch:
        netG.train(); netE.train(); netD.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for i, (images, _) in enumerate(pbar):
            z = torch.randn(batch_size, args.latent_size).to(device)    
            
            generated_image = netG(z).detach()
            generated_latent = netE(images).detach()
            
            reconstructed_image = netE(generated_latent).detach()
            reconstructed_latent = netG(generated_image).detach()
            
            # Model Update
            if i % args.d_iter == 0:
                optD.zero_grad()
                real_score = discriminator([images, generated_latent], training=d_train)  # D(x, E(x))
                fake_score = discriminator([generated_images, latent], training=d_train)  # D(G(z), z)
                
                real_score = netD(images, generated_latent).detach()
                fake_score = netD(generated_image, latent).detach()
                
                d_loss = (fake_score - real_score).mean();
                
                gradient_penalty_loss = gradient_penalty(discriminator,
                                                         images, generated_images,
                                                         latent, generated_latent, device)
                discriminator_total_loss = d_loss + gp_weight * gradient_penalty_loss        
                discriminator_total_loss.backward()
                optD.step()
            else:
                optGE.zero_grad()
                generator_encoder_loss = (real_score - fake_score).mean() # L_E,G
                
                images_reconstruction_loss = (l1(images, reconstructed_images)).mean()  # L_R
                latent_reconstruction_loss = (l1(latent, reconstructed_latent)).mean()  # L_R'
                consistency_loss = images_reconstruction_loss + latent_reconstruction_loss  # L_C
                
                generator_encoder_total_loss = (1 - alpha) * generator_encoder_loss + alpha * consistency_loss  # L*_E,G
                ge_total_loss.backward()
                optGE.step()

            if i % 100 == 0:
                history['d_loss'].append(d_total_loss.item())
                history['ge_loss'].append(ge_total_loss.item())

            pbar.set_postfix({"D": f"{d_total_loss.item():.3f}", "GE": f"{ge_total_loss.item():.3f}"})

        save_reconstruction_grid(images, netE, netG, epoch, device)
        plot_loss_curves(history)
        
        auc = evaluate(netG, netE, netD, test_loader, device, args.lambda_, classes)
        history['auc'].append(auc)
        print(f"Epoch {epoch} | AUC: {auc:.4f}")

        torch.save({
            'generator': generator.state_dict(),
            'encoder': encoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            'config': config
        }, 'final_model.pth')

        pd.DataFrame(history).to_csv('final_training_log.csv', index=False)
        print("Training completed!")

            
def main():    
    class Args:
        image_size = 512
        batch_size = 32
        latent_size = 128
        channels = 3
        lr = 1e-4
        ge_beta1, ge_beta2 = 0.0, 0.1
        d_beta1, d_beta2 = 0.0, 0.9
        gp_weight = 10
        alpha = 1e-5
        d_iter = 1 # Update GE every d_iter steps
        epochs = 50
        lambda_ = 0.1 # Weight for feature distance in scoring
        
        train(Args())
    
