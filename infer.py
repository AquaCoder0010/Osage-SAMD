import torch
from model import *
from SABV import SABV
import cv2
import numpy as np
import torchvision.transforms as transforms

import os

def find_exe_files(root_dir, exe_limit=None):
    exe_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.exe'):
                print(f"found {file}");
                full_path = os.path.join(dirpath, file)
                if exe_limit == None:
                    exe_files.append(full_path)
                elif exe_limit != None and exe_limit >= len(exe_files):
                    exe_files.append(full_path)
                else:
                    break;
    print(f"length : {len(exe_files)}");
    return exe_files



def get_prediction(netG, netE, netD, cv2_img, threshold=0.70, lambda_=0.1):
    device = next(netE.parameters()).device
    
    if cv2_img.shape[2] == 3:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_tensor = transform(cv2_img).unsqueeze(0).to(device)

    netG.eval()
    netE.eval()
    netD.eval()

    with torch.no_grad():
        latent = netE(image_tensor)
        reconstructed_img = netG(latent)

        feat_real = netD(image_tensor, latent, return_features=True)
        feat_recon = netD(reconstructed_img, latent, return_features=True)

        l_r = torch.mean(torch.abs(image_tensor - reconstructed_img))
        
        diff_feat = (feat_real - feat_recon).view(feat_real.size(0), -1)
        l_fd = torch.mean(torch.abs(diff_feat))

        score_val = ((1 - lambda_) * l_r + lambda_ * l_fd).item()

    return {
        "status": "MALWARE" if score_val > threshold else "BENIGN",
        "score": score_val
    }

if __name__ == "__main__":
    class Args:
        image_size = 512
        batch_size = 32
        latent_size = 128
        channels = 3
        lr = 1e-4
        ge_beta1, ge_beta2 = 0.5, 0.999
        d_beta1, d_beta2 = 0.5, 0.999
        gp_weight = 10
        alpha = 0.5
        d_iter = 2
        epochs = 2
        lambda_ = 0.1

    
    checkpoint = torch.load('final_model.pth', map_location='cpu', weights_only=False)
    config = checkpoint['args']
    
    netG = Generator(config.latent_size, config.channels, upsample_first=False, bn_type='batch')
    netE = Encoder(config.image_size, config.latent_size, bn_type='instance')
    netD = Discriminator(config.image_size, config.latent_size, bn_type='layer')

    netG.load_state_dict(checkpoint['generator'])
    netE.load_state_dict(checkpoint['encoder'])
    netD.load_state_dict(checkpoint['discriminator'])

    sabv = SABV(FIS_ENABLED=True, N=5, sample=0.05, FIS_THREADING_ENABLED=True)

    malware_path = os.getcwd() + "/PE-files/DikeDataset-main/files/malware"
    benign_path = os.getcwd() + "/PE-files/DikeDataset-main/files/benign"


    correct = 0;
    total = 0;
    for mal_exe in find_exe_files(malware_path)[1000:1100]:
        img = sabv.process_file(mal_exe)
        results = get_prediction(netG, netE, netD, img)
        print(f" {total + 1} Malware file {mal_exe} is evaluated to be {results["status"]} with score {results["score"]}")
        correct += 1 if results["status"] == "MALWARE" else 0
        total += 1

    print(f"accuracy : {correct / total}")
    pass
