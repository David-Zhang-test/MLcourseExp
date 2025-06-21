import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from trainer import Trainer

from model import Classifyer

args = Namespace(
    num_epochs = 15,
    batch_size = 64,
    lr = 0.001,
    num_channels=10,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mean and std for each channel
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



# model and criterion
model = Classifyer(args.num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


trainer = Trainer(
    dataloader={'train': train_loader, 'test': test_loader},
    model=model,
    checkpoint='checkpoint.pth',
    save_dir='./saved_dir',
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    early_stopping_criteria=5,
    device = device
)


trainer.run_train_loop()
trainer.run_test_loop()
trainer.plot_performance()

