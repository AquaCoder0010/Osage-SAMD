import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Scales to [-1, 1] for Tanh
    ])

    # Train: Expects ./train/some_class/image.png
    train_dataset = datasets.ImageFolder(root='./dataset/train', transform=transform)

    if 'benign' in train_dataset.classes:
        benign_class_idx = train_dataset.class_to_idx['benign']
        
        # Filter samples to include only those belonging to the 'benign' class
        train_dataset.samples = [
            (path, label) for path, label in train_dataset.samples
            if label == benign_class_idx
        ]
        # The new class list for the train set is now just ['benign']
        print(f"Training on {len(train_dataset.samples)} samples from the 'benign' class.")
    else:
        # If 'benign' is not found, the FileNotFoundError was probably correct.
        raise FileNotFoundError(f"Could not find the expected 'benign' class folder inside {train_root_path}")
    
    # Test: Expects ./test/benign/ and ./test/malware/
    test_dataset = datasets.ImageFolder(root='./dataset/test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, test_dataset.classes
