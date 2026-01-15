import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

import os
import time

wandb.login(key= "USE-YOUR-OWN-PRIVATE-KEY")

# =====================
# (1) TRUE 2012 ALEXNET
# =====================

class AlexNet_2012(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_2012, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

        # Weight init EXACTLY as in paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =======================================
# (2) DATA PIPELINE — CIFAR-10 → 224×224
# =======================================

def get_dataloaders(batch_size=128):

    transform_train = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    trainset = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=transform_train,)
    testset = datasets.CIFAR10(root="./data", train=False,
                               download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    return trainloader, testloader

from tqdm import tqdm
import wandb

def train_with_wandb_tqdm(model, optimizer, scheduler, criterion, train_loader, val_loader,
                          device, epochs=50, grad_clip=5.0, val_steps=100):
    """
    Training loop for AlexNet with:
      - Gradient clipping
      - Cosine LR
      - WandB logging
      - Validation every `val_steps` batches
      - TQDM progress bar
    """
    wandb.init(
        project="Alex-Net",
        name = "alex-net-step-1",
        config={
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": "AlexNet-2012",
            "dataset": "CIFAR-10",
            "batch_size": train_loader.batch_size,
            "epochs": epochs
        }
    )

    def evaluate(model, loader, device):
      model.eval()
      correct = 0
      total = 0

      with torch.no_grad():
          for imgs, labels in loader:
              imgs, labels = imgs.to(device), labels.to(device)
              outputs = model(imgs)
              preds = outputs.argmax(dim=1)
              correct += (preds == labels).sum().item()
              total += labels.size(0)

      return correct / total


    best_acc = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (imgs, labels) in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            global_step += 1

            # Perform validation every val_steps
            if global_step % val_steps == 0:
                val_acc = evaluate(model, val_loader, device)
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for v_imgs, v_labels in val_loader:
                        v_imgs, v_labels = v_imgs.to(device), v_labels.to(device)
                        v_outputs = model(v_imgs)
                        v_loss = criterion(v_outputs, v_labels)
                        val_loss += v_loss.item() * v_imgs.size(0)
                val_loss /= len(val_loader.dataset)
                model.train()

                wandb.log({
                    "train_loss": running_loss / running_total,
                    "train_acc": 100 * running_correct / running_total,
                    "val_loss": val_loss,
                    "val_acc": 100 * val_acc,
                    "epoch": epoch + batch_idx / len(train_loader),
                    "step": global_step
                })

                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), "alexnet2012_cifar10_best.pth")

                # Reset running metrics for next interval
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            # Update tqdm postfix
            loop.set_postfix({
                "train_loss": f"{running_loss/running_total:.4f}" if running_total > 0 else "0.0000",
                "train_acc": f"{100*running_correct/running_total:.2f}%" if running_total > 0 else "0.00%"
            })

        # Optional: epoch-level validation logging
        epoch_val_acc = evaluate(model, val_loader, device)
        wandb.log({
            "epoch": epoch + 1,
            "epoch_val_acc": 100 * epoch_val_acc
        })

    print(f"Training complete. Best validation accuracy: {100*best_acc:.2f}%")


if __name__ == "__main__":
  # Initialize your model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = AlexNet_2012(num_classes=10).to(device)

  # Define loss
  criterion = nn.CrossEntropyLoss()

  # Define optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

  # Get dataloaders
  train_loader, val_loader = get_dataloaders(batch_size=128)

  # Define cosine annealing scheduler
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=13 * len(train_loader))

  # Call the training function
  train_with_wandb_tqdm(
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      train_loader=train_loader,
      val_loader=val_loader,
      device=device,
      epochs=13,
      grad_clip=1.0,
      val_steps=100
  )

