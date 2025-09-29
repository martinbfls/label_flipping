from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch

def get_targeted_data(loader, target_label, max_samples=1000, batch_size=64):
    collected_data = []
    collected_targets = []
    total_collected = 0
    for data, target in loader:
        idx = (target == target_label).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            remaining_needed = max_samples - total_collected
            idx = idx[:remaining_needed]
            
            collected_data.append(data[idx])
            collected_targets.append(target[idx])
            total_collected += len(idx)
            
            if total_collected >= max_samples:
                break
        
    if total_collected == 0:
        raise ValueError(f"No data found for target label {target_label}")
    
    all_data = torch.cat(collected_data, dim=0)
    all_targets = torch.cat(collected_targets, dim=0)
    
    dataset = TensorDataset(all_data, all_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_mnist(train_size=10000, test_size=10000, batch_size=64, test_batch_size=1000, target_label=5, targeted_data_size=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    remaining_train_size = len(train_dataset) - train_size
    train_set, remaining_set = random_split(train_dataset, [train_size, remaining_train_size])
    
    test_dataset = datasets.MNIST('.', train=False, transform=transform)
    actual_test_size = min(test_size, len(test_dataset))
    if actual_test_size < len(test_dataset):
        test_set, _ = random_split(test_dataset, [actual_test_size, len(test_dataset) - actual_test_size])
    else:
        test_set = test_dataset

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    targeted_loader = get_targeted_data(DataLoader(remaining_set, batch_size=batch_size, shuffle=True), target_label, targeted_data_size, batch_size)
    return train_loader, test_loader, targeted_loader

def load_mnist_trajectory_matching(train_size=10000, test_size=10000, batch_size=64, 
                                    test_batch_size=1000, target_label=5, adversarial_label=3):

    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    remaining_train_size = len(train_dataset) - train_size
    
    train_set, _ = random_split(train_dataset, [train_size, remaining_train_size])
    
    test_dataset = datasets.MNIST('.', train=False, transform=transform)
    actual_test_size = min(test_size, len(test_dataset))
    if actual_test_size < len(test_dataset):
        test_set, _ = random_split(test_dataset, [actual_test_size, len(test_dataset) - actual_test_size])
    else:
        test_set = test_dataset
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    
    poisoned_data = []
    for img, label in train_set:
        if label == target_label:
            poisoned_data.append((img, adversarial_label))
        else:
            poisoned_data.append((img, label))
    poisoned_loader = DataLoader(poisoned_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, poisoned_loader
