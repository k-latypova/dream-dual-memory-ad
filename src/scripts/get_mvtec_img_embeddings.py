import argparse
import numpy as np
import torch
from src.trainers.pretrain_mvtec import get_class_embeddings
from src.datasets.mvtec_dataset import MVTecDataset
from torchvision.transforms import transforms as T
import json
from src.networks.resnet_anchor import ResNet_Model
import os
from src.trainers.mvtec_ad_pretrainer import MvtecADPreTrainer


def get_embeddings(args):
    pretrainer = MvtecADPreTrainer(args)
    embeddings = pretrainer.get_class_embeddings()
    os.path.join(args.output_dir, "mvtec_embeddings_npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get image embeddings")
    parser.add_argument("--mvtec-root", type=str, required=True, help="Path to the MVTec AD dataset directory")
    parser.add_argument("--mvtec2-root", type=str, required=True, help="Path to the MVTec2 AD dataset directory")
    parser.add_argument("--imagenet-root", type=str, required=True, help="Path to the ImageNet dataset directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--width", type=int, default=224, help="Image width for training")
    parser.add_argument("--height", type=int, default=224, help="Image height for training")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save the model and embeddings")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for image augmentation")
    parser.add_argument("--load_weights", type=str, default=None, help="Weights to contrinue training from")
    parser.add_argument("--text_encoders_file", type=str, required=True, help="Path to the files to match class categories with text encoder weights")
    args = parser.parse_args()
    args.lr = 1e-4
    args.momentum = 1e-4
    args.weight_decay=1e-4
    args.epochs=100
    args.init_epoch=0
    get_embeddings(args)

