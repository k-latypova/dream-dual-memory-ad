import json
import os

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import random_split
from tqdm.auto import tqdm

from src.networks.cnn import CNN32
from src.utils.training_utils import fpr_and_fdr_at_recall
from src.datasets.cifar_dataset import CifarDataset
from src.datasets.imagenet_dataset import Imagenet100Dataset


class ADTrainer:
    def __init__(self, args):
        self.device = args.device
        self.args = args
        self.__init_seed()
        self.__init_model()
        self.__init_dataset()
        self.__init_training_utils()
        self.__init_logs()



    def __init_model(self):
        args = self.args
        self.model = CNN32(rep_dim=512, bias=True, clf=False, grayscale=False)
        self.model = self.model.to(args.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()


    def __init_dataset(self):
        args = self.args
        def test_label_transform(x):
            return 0 if x == args.normal_class else 1
        
        def train_label_transform(x):
            return x if x <= args.normal_class else x - 1
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)

        mean = stats[str(args.normal_class)]['mean']
        std = stats[str(args.normal_class)]['std']
        print(f"Mean: {mean}, Std: {std}")
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        # if args.in_dst == 'cifar10':
        #     dst_cls = dset.CIFAR10
        # elif args.in_dst == 'cifar100':
        #     dst_cls = dset.CIFAR100
        # else:
        #     raise ValueError(f"Unknown dataset: {args.dst}")
        # train_data_in = dst_cls(os.path.join(args.data_dir, args.in_dst), download=True, train=True, 
        #                         transform=train_transform,
        #                         target_transform=trn.Lambda(lambda x: train_label_transform(x)))
      
        # normal_class = self.args.normal_class
        # target_normald_idx = np.argwhere(np.array(train_data_in.targets) == normal_class).flatten()
        # train_data_in = torch.utils.data.Subset(train_data_in, target_normald_idx)
        if args.in_dst.startswith('cifar'):
            train_data_in = CifarDataset(args.in_dst, args.data_dir, args.normal_class, train=True, 
                                         transform=train_transform)
            print(f"Found {len(train_data_in)} training samples in {args.in_dst} dataset")
            test_dataset = CifarDataset(args.in_dst, args.data_dir, args.normal_class, train=False,
                                         transform=test_transform)
        elif args.in_dst.startswith('imagenet'):
            train_data_in = Imagenet100Dataset(args.in_dst, args.data_dir, args.normal_class, train=True, 
                                            transform=train_transform)
            test_dataset = Imagenet100Dataset(args.in_dst, args.data_dir, args.normal_class, train=False,
                                            transform=test_transform)
        else:
            raise ValueError(f"Unknown dataset: {args.in_dst}")
        train_data_length = 0.9 * len(train_data_in)
        val_data_length = len(train_data_in) - train_data_length
        train_data_in, val_data_in = random_split(train_data_in, [int(train_data_length), int(val_data_length)])
        trainloader = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, 
                                                  num_workers=self.args.num_workers, pin_memory=True,
                                                  prefetch_factor=2)
        valloader = torch.utils.data.DataLoader(val_data_in, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=self.args.num_workers)
        
        ood_data = dset.ImageFolder(args.ad_data_dir, transform=train_transform)
        print(f"Found {len(ood_data)} OOD samples in {args.ad_data_dir}")

        train_ood, val_ood = random_split(ood_data, [int(0.9 * len(ood_data)), len(ood_data) - int(0.9 * len(ood_data))])

        train_ood_dataloader = torch.utils.data.DataLoader(train_ood, batch_size=args.ad_batch_size, shuffle=False,
                                                           num_workers=self.args.num_workers, pin_memory=True, 
                                                           prefetch_factor=2)
        
        val_ood_dataloader = torch.utils.data.DataLoader(val_ood, batch_size=args.ad_batch_size, shuffle=False,
                                                         num_workers=self.args.num_workers, pin_memory=True, 
                                                         prefetch_factor=2)

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                                 num_workers=self.args.num_workers)
        
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.train_ood_dataloader = train_ood_dataloader
        self.val_ood_dataloader = val_ood_dataloader


    def __init_logs(self):
        logs_dir = self.args.save
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir
        with open(os.path.join(logs_dir, 'args.json'), 'w') as f:
            json.dump(vars(self.args), f)
        with open(os.path.join(logs_dir, 'training.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss,val_auc\n')
        

    def __init_training_utils(self):

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                       milestones=[10, 15, 20], gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                                                    patience=5, verbose=True)


    def compute_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        dists = torch.sqrt(torch.norm(features, p=2, dim=1) ** 2 + 1) - 1
        scores = (1 - torch.exp(-dists))
        return scores

        
    def __init_seed(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.args.seed)

    def loss(self, features, labels, nominal_label=0):
        dists = torch.sqrt(torch.norm(features, p=2, dim=1) ** 2 + 1) - 1
        scores = 1 - torch.exp(-dists)
        losses = torch.where(labels == nominal_label, dists, -torch.log(scores + 1e-9))
        return losses.mean()

    def train_epoch(self):
        self.model.train()
        avg_loss = 0.0
        self.train_ood_dataloader.dataset.offset = np.random.randint(len(self.train_ood_dataloader.dataset))
        for in_set, out_set in zip(self.trainloader, self.train_ood_dataloader):
            in_data, in_target = in_set
            in_target = torch.zeros(in_data.size(0)).to(self.device)
            out_data, _ = out_set
            data = torch.cat([in_data, out_data], dim=0).to(self.device)

            permutation_index = torch.randperm(data.size(0))
            fake_target = torch.cat([in_target, torch.ones(out_data.size(0)).to(self.device)], dim=-1)
            data = data[permutation_index]

            output = self.model(data)
            self.optimizer.zero_grad()
            print(f"Output.shape: {output.shape}, Fake target shape: {fake_target[permutation_index].shape}")
            
            loss = self.loss(output, fake_target[permutation_index])

            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        avg_loss = avg_loss / len(self.trainloader)
        self.scheduler.step(avg_loss)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_scores = []
        ds_targets = []
        with torch.no_grad():
            for in_data, ood_data in zip(self.valloader, self.val_ood_dataloader):
                data, target = in_data[0].to(self.device), in_data[1].to(self.device)
                ood_data, _ = ood_data
                ood_data = ood_data.to(self.device)
                target = torch.zeros(data.size(0)).long().to(self.device)
                all_data = torch.cat([data, ood_data.to(self.device)], dim=0)
                all_targets = torch.cat([target, torch.ones(ood_data.size(0)).long().to(self.device)], dim=0)
                output = self.model(all_data)
                loss = self.loss(output, all_targets)
                val_loss += loss.item()
                scores = self.compute_anomaly_score(output)
                scores = scores.cpu().numpy()
                all_scores.append(scores)
                ds_targets.append(all_targets.cpu().numpy())

        val_auc = roc_auc_score(np.concatenate(ds_targets), np.concatenate(all_scores))

        length = len(self.valloader)
        val_loss /= length

        return val_loss, val_auc
    
    def train(self):
        args = self.args
        best_val_loss = 1e15
        best_epoch = -1
        self.validate()
        for epoch in tqdm(range(args.epochs)):
            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate()
            with open(os.path.join(self.logs_dir, 'training.csv'), 'a') as f:
                f.write(f'{epoch},{train_loss},{val_loss},{val_auc}\n')
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}')
            torch.save({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch
            }, os.path.join(self.logs_dir, f'model_{epoch}.pth'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch
                }, os.path.join(self.logs_dir, 'best_model.pth'))

        print(f'Best Val Loss: {best_val_loss} on epoch {best_epoch}')
                   

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.logs_dir, 'best_model.pth'), map_location=self.device)['state_dict'])
        #self.train()
        self.model.eval()
        score = []
        targets = np.array([])
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                scores = self.compute_anomaly_score(output)
                score.append(scores.cpu().numpy())

                targets = np.concatenate((targets, target.cpu().numpy()))
        score = np.concatenate(score)
        auc = roc_auc_score(targets, score)
        print(f'AUC: {auc}')
        aupr = average_precision_score(targets, score)
        print(f'AUPR: {aupr}')
        fpr = fpr_and_fdr_at_recall(targets, score, 0.95)
        print(f'FPR: {fpr}')
        precision, recall, thresholds = precision_recall_curve(targets, score)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_f1 = np.max(f1)
        best_f1_threshold = thresholds[np.argmax(f1)]
        print(f'Best F1: {best_f1} with threshold: {best_f1_threshold}')

        with open(os.path.join(self.logs_dir, 'test_results.json'), 'w') as f:
            json.dump({'auc': float(auc), 
                       'aupr': float(aupr), 
                       'fpr': float(fpr), 
                       'f1': float(best_f1), 
                       'best_threshold': float(best_f1_threshold)}, f)