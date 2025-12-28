import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# A simple custom dataset to load all images in a flat folder
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # List all files and filter for common image extensions
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return image and a dummy label (0) to keep it compatible with training loops
        return image, 0

def get_dataloaders(image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Scales to [-1, 1] for Tanh
    ])

    # NEW: Load all images directly from the train folder
    train_root = './dataset/train'
    train_dataset = FlatImageDataset(root_dir=train_root, transform=transform)

    print(f"Training on {len(train_dataset)} samples found in {train_root}.")

    # Test: Keep this as ImageFolder if your test set still uses benign/malware subfolders
    # If test is also flat, use FlatImageDataset for it as well.
    test_dataset = datasets.ImageFolder(root='./dataset/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, test_dataset.classes