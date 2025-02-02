import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import argparse
import os
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pneumonia Detection Training')
    parser.add_argument('--data_dir', type=str, default='chest_xray', help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size for training')
    parser.add_argument('--epochs_base', type=int, default=10, help='Number of base training epochs')
    parser.add_argument('--epochs_ft1', type=int, default=8, help='Number of first fine-tuning epochs')
    parser.add_argument('--epochs_ft2', type=int, default=15, help='Number of final fine-tuning epochs')
    parser.add_argument('--output_dir', type=str, default='model', help='Output directory for plots and model')
    return parser.parse_args()

def prepare_data(data_dir, batch_size):
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    test_dir = f"{data_dir}/test"
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(f"{data_dir}/train", data_transforms['train']),
        'val': datasets.ImageFolder(f"{data_dir}/val", data_transforms['val']),
        'test': datasets.ImageFolder(f"{data_dir}/test", data_transforms['val'])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                      shuffle=True if x == 'train' else False, num_workers=4)
        for x in ['train', 'val', 'test']
    }

    return image_datasets, dataloaders

def initialize_model(device):
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    return model

def train_model(model, criterion, optimizer, dataloaders, image_datasets, 
               scheduler=None, num_epochs=10, phase_name=''):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} {phase_name}'.strip())
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == 'train' and scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    pass  
                else:
                    scheduler.step()

            if phase == 'val' and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_acc)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch complete in {epoch_time:.0f}s\n')

    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss, train_acc, val_acc

def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    image_datasets, dataloaders = prepare_data(args.data_dir, args.batch_size)
    
    # Initialize model
    model = initialize_model(device)
    criterion = nn.CrossEntropyLoss()
    
    # Base training phase
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, criterion, optimizer, dataloaders, image_datasets, num_epochs=args.epochs_base
    )
    
    # First fine-tuning phase
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name:
            param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model, train_loss_ft1, val_loss_ft1, train_acc_ft1, val_acc_ft1 = train_model(
        model, criterion, optimizer, dataloaders, image_datasets, scheduler, args.epochs_ft1, 'FT1'
    )
    
    # Final fine-tuning phase
    for param in model.parameters():
        param.requires_grad = True
    optimizer_ft = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='max', patience=2, factor=0.5, min_lr=1e-7, verbose=True
    )
    model, train_loss_ft2, val_loss_ft2, train_acc_ft2, val_acc_ft2 = train_model(
        model, criterion, optimizer_ft, dataloaders, image_datasets, scheduler_ft, args.epochs_ft2, 'FT2'
    )
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{args.output_dir}/pneumonia_model.pth')

if __name__ == '__main__':
    main()