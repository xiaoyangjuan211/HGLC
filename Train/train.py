# Train/train.py
import os
import sys
import torch
import torch.nn as nn
import argparse
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import convnext_tiny_ultimate
from Loss.loss import CrossEntropyLabelSmooth


def get_transform(dataset_name):
    if dataset_name == "JAFFE":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 加载脱敏后的模型
    model = convnext_tiny_ultimate(num_classes=args.num_classes).to(device)

    # 2. 损失函数
    criterion = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=args.label_smoothing)

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Starting training DGLNet on {args.dataset}...")
    # 模拟训练过程
    for epoch in range(args.epochs):
        # train_one_epoch(model, None, criterion, optimizer, device) # 占位
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {0.5 / (epoch + 1):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    main(args)