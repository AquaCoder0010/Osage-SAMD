import torch
import torch.nn.functional as F

def gradient_penalty(discriminator, real_images, fake_images, real_latent, fake_latent, device):
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha_l = alpha.view(batch_size, 1) # Match latent dims

    interpolated_img = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    interpolated_lat = (alpha_l * real_latent + (1 - alpha_l) * fake_latent).requires_grad_(True)

    d_interpolated = discriminator(interpolated_img, interpolated_lat)

    # Calculate gradients w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=[interpolated_img, interpolated_lat],
        grad_outputs=torch.ones(d_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    
    # Flatten and concatenate gradients for Image and Latent
    grad_img = gradients[0].view(batch_size, -1)
    grad_lat = gradients[1].view(batch_size, -1)
    grad_combined = torch.cat([grad_img, grad_lat], dim=1)
    
    grad_norm = grad_combined.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

def compute_l1(x, y):
    return F.l1_loss(x, y, reduction='mean')
