import argparse
import numpy as np
import torch
from src.trainers.pretrain_mvtec import get_class_embeddings
from src.datasets.mvtec_dataset import MVTecDataset
from torchvision.transforms import transforms as T
import json
from src.networks.resnet_anchor import ResNet_Model
import os


def get_embeddings(args):
    dataset = MVTecDataset(root=args.data_dir,
                           class_name=args.class_name,
                           split='train',
                           transform=T.Compose([T.Resize((args.img_size, args.img_size)),
                                               T.ToTensor(),
                                               T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
                           )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    save_path = os.path.dirname(args.anchor_file)
    anchor_filename = args.anchor_file
    anchor = torch.from_numpy(np.load(anchor_filename)).to(args.device)
    anchor = anchor.reshape(anchor.size(0), 768)
    num_classes = anchor.size(0)
    model = ResNet_Model(num_classes=num_classes).to(args.device)
    if args.weights_path:
        print(f"Loading weights from {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    model.eval()
    embeddings = get_class_embeddings(model, train_loader, args.device)
    save_filename = os.path.join(save_path, f"{args.class_name}_embeddings.npy")
    np.save(save_filename, embeddings)
    print(f"Embeddings saved to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image embeddings")
    parser.add_argument('--data-dir', type=str, default='data/mvtec', help='Directory containing the MVTec dataset')
    parser.add_argument('--class-name', type=str, default='bottle', help='Name of the class to pretrain on')
    parser.add_argument('--anchor-file', type=str, default='token_embed_mvtec.npy', help='Path to the anchor file')
    parser.add_argument('--stats-file', type=str, default='stats_mvtec.json', help='Path to the stats file')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--weights-path', type=str, help='Path to the pretrained weights file')
    args = parser.parse_args()
    
    get_embeddings(args)