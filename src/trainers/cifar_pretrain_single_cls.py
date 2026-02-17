import numpy as np
import os

import argparse
import torch

import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch.nn.functional as F

from src.networks.resnet_anchor import ResNet_Model
from src.utils.training_utils import cosine_annealing

from tqdm import tqdm
import json

def pretrain_cifar(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    # save_path = os.path.join(args.save_dir, args.dataset, str(args.normal_class))
    save_path = os.path.dirname(args.anchor_file)
    os.makedirs(save_path, exist_ok=True)
    snapshots = os.path.join(save_path, "pretrain")
    os.makedirs(snapshots, exist_ok=True)

    
    with open(args.stats_file, 'r') as f:
        stats = json.load(f)

    mean = stats[str(args.normal_class)]['mean']
    std = stats[str(args.normal_class)]['std']
    print(f"Mean: {mean}, Std: {std}")
    # Load CIFAR-10 dataset
    if args.dataset == 'cifar10':
        dst_cls = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        dst_cls = datasets.CIFAR100
    data_dir = os.path.join(args.data_dir, args.dataset)
    dataset = dst_cls(
        root=data_dir,
        train=True,
        download=True,
        transform=T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    )
    class_indices = np.where(np.array(dataset.targets) == args.normal_class)[0]
    train_dataset = torch.utils.data.Subset(dataset, class_indices)
    print(f"Number of samples in the subset: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    anchor_filename = args.anchor_file
    anchor = torch.from_numpy(np.load(anchor_filename)).to(args.device)
    anchor = anchor.reshape(anchor.size(0), 768)
    num_classes = anchor.size(0)

    # Initialize model
    model = ResNet_Model(num_classes=num_classes).to(args.device)
    model.train()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr))


    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(args.device), target.to(args.device)
            target = torch.zeros(data.size(0), dtype=torch.long).to(args.device)  # All targets are set to 0 (normal class)

            optimizer.zero_grad()
            embeddings = model(data)
            dists = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(embeddings), 1, 1),
                                        embeddings.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1

            loss = criterion(dists, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(train_loader)}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(snapshots, f"model_epoch_{epoch + 1}.pth"))

    embeddings = get_class_embeddings(model, train_loader, args.device)
    np.save(os.path.join(save_path, f"embeddings_epoch_{epoch + 1}_{args.dataset}.npy"), embeddings)


def get_class_embeddings(model, data_loader, device):
    """Get class embeddings from the model.

    Args:
        model (torch.nn.Module): Model to get embeddings from.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Class embeddings.
    """
    model.eval()
    #embeddings_counts = {x: 0 for x in range(num_classes)}
    embeddings = torch.zeros(500, 768).to(device)  # Assuming 768 is the embedding size
    index = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(data_loader):
            data = data.to(device)
            print(f"Data shape: {data.shape}")
            features = model(data)
            for i in range(data.shape[0]):
                if index >= 500:
                    break
                embeddings[index] = features[i]
                index += 1
            if index >= 500:
                break

    return embeddings.cpu().numpy()
