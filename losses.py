import torch
import torch.nn as nn
import torch.autograd as autograd

def l1(x, y):
    """
    Computes L1 distance keeping batch dimension.
    """
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    return torch.sum(torch.abs(x - y), dim=1)

def gradient_penalty(discriminator, x, x_gen, z, z_gen, device):
    """
    Calculates the WGAN-GP gradient penalty.
    """
    batch_size = x.size(0)
    
    alpha = torch.rand(batch_size, 1, device=device)
    alpha_img = alpha.view(batch_size, 1, 1, 1) 
    alpha_z = alpha.view(batch_size, 1)

    x_hat = (alpha_img * x + (1 - alpha_img) * x_gen).detach().requires_grad_(True)
    z_hat = (alpha_z * z + (1 - alpha_z) * z_gen).detach().requires_grad_(True)
    score_hat = discriminator(x_hat, z_hat)

    gradients = autograd.grad(
        outputs=score_hat,
        inputs=[x_hat, z_hat],
        grad_outputs=torch.ones_like(score_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    
    dx, dz = gradients
    
    dx = dx.view(dx.size(0), -1)
    dz = dz.view(dz.size(0), -1)
    
    grads = torch.cat([dx, dz], dim=1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)    
    norm_penalty = (grads_norm - 1) ** 2
    return norm_penalty.mean()
