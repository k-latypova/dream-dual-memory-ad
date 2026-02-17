import json
import os

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T

from src.datasets.mvtec_dataset import MVTecDataset
from src.networks.resnet import ResNet_Model
#from src.networks.resnet_1 import resnet18_enc_dec
from src.utils.training_utils import fpr_and_fdr_at_recall
import pandas as pd
import cv2
from skimage import measure
from sklearn import metrics
from PIL import Image
from tqdm.auto import tqdm
from src.mvtec_ad_evaluation.evaluate_experiment import calculate_au_pro_au_roc

MVTEC_IMAGE_SIZE = 1024 # Default image size for MVTec dataset, can be adjusted based on the dataset used

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        if not self.args.test_batch_size:
            self.args.test_batch_size = self.args.batch_size
        self.__init_seed()
        self.__init__dataset()
        self.__init__model()
        self.__init_optimizers()
        self.__init_logs()
        self.unfold = torch.nn.Unfold(self.args.patch_size, dilation=1, padding=1, stride=self.args.patch_stride)


    def __init__dataset(self):
        with open(self.args.stats_file, "r") as f:
            stats = json.load(f)
        self.stats = stats[self.args.class_name]
        mean = np.array(self.stats['mean']).flatten() / 255.0
        std = np.array(self.stats['std']).flatten() / 255.0

        train_transform = T.Compose([
            T.Resize((self.args.image_size, self.args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)]
        )
        test_transform = T.Compose([
            #T.Resize((self.args.image_size, self.args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)]
        )
        self.train_dataset = MVTecDataset(
            root=self.args.data_path,
            class_name=self.args.class_name,
            split='train',
            transform=train_transform
        )

        self.outliers_dataset = ImageFolder(
            root=self.args.outliers_path,
            transform=train_transform
        )
        if self.args.num_outliers is None:
            self.args.num_outliers = len(self.outliers_dataset)
        outliers_idx = np.random.choice(
            len(self.outliers_dataset),
            size=self.args.num_outliers,
            replace=False
        )
        self.outliers_dataset = torch.utils.data.Subset(self.outliers_dataset, outliers_idx)
        train_length = int(len(self.train_dataset) * self.args.train_split)
        val_length = len(self.train_dataset) - train_length
        self.train_subset, self.val_subset = random_split(
            self.train_dataset,
            [train_length, val_length]
        )
        train_outliers_length = int(len(self.outliers_dataset) * self.args.train_split)
        val_outliers_length = len(self.outliers_dataset) - train_outliers_length
        self.outliers_train_subset, self.outliers_val_subset = random_split(
            self.outliers_dataset,
            [train_outliers_length, val_outliers_length]
        )
        self.train_outliers_loader = torch.utils.data.DataLoader(
            self.outliers_train_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.val_outliers_loader = torch.utils.data.DataLoader(
            self.outliers_val_subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.test_dataset = MVTecDataset(
            root=self.args.data_path,
            class_name=self.args.class_name,
            split='test',
            transform=test_transform,
            ground_truth_transform=T.Compose([T.Resize((self.args.image_size, self.args.image_size)), T.ToTensor()])
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def __init__model(self):
        self.model = ResNet_Model(self.args.resnet, 1)
        #self.model = resnet18_enc_dec(num_classes=1, pool=True, final_activation="sigmoid", preact=False)
        self.model = self.model.to(self.args.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        #self.criterion = torch.nn.BCELoss()

    def __init_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=5,
                                                                    verbose=True)
        
    def __init_seed(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.args.seed)

    def __init_logs(self):
        logs_dir = self.args.save
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir
        with open(os.path.join(logs_dir, 'args.json'), 'w') as f:
            json.dump(vars(self.args), f)
        with open(os.path.join(logs_dir, 'training.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss,val_auc\n')

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch, outliers in zip(self.train_loader, self.train_outliers_loader):
            images, _ = batch
            outliers, _ = outliers
            images = torch.cat((images, outliers), dim=0)
            targets = torch.cat((torch.zeros(len(batch[0]), 1), torch.ones(len(outliers), 1)), dim=0).to(self.args.device)
            images = images.to(self.args.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_scores = []
        all_targets = []
        with torch.no_grad():
            for batch, outliers_batch in zip(self.val_loader, self.val_outliers_loader):
                images, _ = batch
                outliers, _ = outliers_batch
                images = torch.cat((images, outliers), dim=0)
                targets = torch.cat((torch.zeros(len(batch[0]), 1), torch.ones(len(outliers), 1)), dim=0).to(self.args.device)
                images = images.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                #scores = torch.mean(outputs, dim=(1, 2, 3))

                all_scores.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        auc_score = roc_auc_score(all_targets, all_scores)

        return avg_loss, auc_score
      
    def train(self):
        for epoch in tqdm(range(self.args.epochs)):
            self.validate(epoch)
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate(epoch)
            self.scheduler.step(val_loss)
            self.log_results(epoch, train_loss, val_loss, val_auc)
            torch.save(self.model.state_dict(), os.path.join(self.logs_dir, f'model_epoch_{epoch}.pth'))

    def log_results(self, epoch, train_loss, val_loss, val_auc):
        log_file = os.path.join(self.logs_dir, 'training.csv')
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss},{val_auc}\n")
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}")

    def test(self):
        #self.model.load_state_dict(torch.load(os.path.join(self.logs_dir, 'best_model.pth'), map_location=self.device)['state_dict'])
        os.makedirs(os.path.join(self.logs_dir, 'anomaly_maps'), exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, 'anomaly_maps_gt'), exist_ok=True)

        self.model.eval()
        score = []
        targets = np.array([])
        anomaly_maps = []
        ground_truths = []
        print("Starting testing...")
        idx = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                data, target, ground_truth = batch
                data, target, ground_truth = data.to(self.device), target.to(self.device), ground_truth.to(self.device)
                output = self.model(data)
                scores = self.compute_anomaly_score(output)
                score.append(scores.cpu().numpy())
                anomaly_map = self.compute_anomaly_map(data)
                anomaly_map = anomaly_map.cpu().numpy().reshape(data.size(0), MVTEC_IMAGE_SIZE, MVTEC_IMAGE_SIZE)
                for anomaly_mask, anomaly_mask_gt in zip(anomaly_map, ground_truth):
                    anomaly_map_path = os.path.join(self.logs_dir, 'anomaly_maps', str(idx))
                    anomaly_mask_img = Image.fromarray(anomaly_mask * 255).convert('L')
                    anomaly_mask_img.save(anomaly_map_path + '.tiff', format='TIFF')
                    ground_truth_path = os.path.join(self.logs_dir, 'anomaly_maps_gt', str(idx))
                    ground_truth_img = Image.fromarray(anomaly_mask_gt.cpu().numpy().reshape(MVTEC_IMAGE_SIZE, MVTEC_IMAGE_SIZE) * 255).convert('L')
                    ground_truth_img.save(ground_truth_path + '.tiff', format='TIFF')
                    anomaly_maps.append(anomaly_map_path)
                    ground_truths.append(ground_truth_path)
                    idx += 1 
                targets = np.concatenate((targets, target.cpu().numpy()))

                # with open(os.path.join(self.logs_dir, 'anomaly_maps', f'{idx}_ground_truth.png'), 'wb') as f:
                #     ground_truth_img = Image.fromarray(ground_truth[0].reshape(MVTEC_IMAGE_SIZE, MVTEC_IMAGE_SIZE).cpu().numpy() * 255).convert('L')
                #     ground_truth_img.save(f, format='PNG')
                # with open(os.path.join(self.logs_dir, 'anomaly_maps', f'{idx}_map.png'), 'wb') as f:
                #     map_img = Image.fromarray(anomaly_map[0].reshape(MVTEC_IMAGE_SIZE, MVTEC_IMAGE_SIZE) * 255).convert('L')
                #     map_img.save(f, format='PNG')

        au_pro, au_roc, _, _ = calculate_au_pro_au_roc(ground_truths, anomaly_maps, integration_limit=0.3)

        score = np.concatenate(score)
        true_labels = (targets != 0).astype(np.float32)
        auc = roc_auc_score(true_labels, score)
        aupr = average_precision_score(true_labels, score)
        print(f'AUPR: {aupr}')
        fpr = fpr_and_fdr_at_recall(true_labels, score, 0.95)
        print(f'FPR: {fpr}')
        precision, recall, thresholds = precision_recall_curve(true_labels, score)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_f1 = np.max(f1)
        best_f1_threshold = thresholds[np.argmax(f1)]
        print(f'Best F1: {best_f1} with threshold: {best_f1_threshold}')

        with open(os.path.join(self.logs_dir, 'test_results.json'), 'w') as f:
            json.dump({'auc': float(auc), 
                       'aupr': float(aupr), 
                       'fpr': float(fpr), 
                       'f1': float(best_f1), 
                       'au_pro': float(au_pro),
                        'au_roc': float(au_roc),
                       #'pro_auc': float(pro_auc),
                       'best_threshold': float(best_f1_threshold)}, f)


    # def compute_pro(self, masks, amaps, num_th=200):
    #     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    #     binary_amaps = np.zeros_like(amaps, dtype=bool)

    #     min_th = amaps.min()
    #     max_th = amaps.max()
    #     delta = (max_th - min_th) / num_th

    #     k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     for th in np.arange(min_th, max_th, delta):
    #         binary_amaps[amaps <= th] = 0
    #         binary_amaps[amaps > th] = 1

    #         pros = []
    #         for binary_amap, mask in zip(binary_amaps, masks):
    #             binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
    #             for region in measure.regionprops(measure.label(mask)):
    #                 axes0_ids = region.coords[:, 0]
    #                 axes1_ids = region.coords[:, 1]
    #                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
    #                 pros.append(tp_pixels / region.area)

    #         inverse_masks = 1 - masks
    #         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
    #         fpr = fp_pixels / inverse_masks.sum()

    #         df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

    #     df = df[df["fpr"] < 0.3]
    #     df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    #     pro_auc = metrics.auc(df["fpr"], df["pro"])
    #     return pro_auc

    
        
    def compute_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        # dists = torch.sqrt(torch.norm(features, p=2, dim=1) ** 2 + 1) - 1
        # scores = (1 - torch.exp(-dists))
        # return scores
        return features
        
        
        

