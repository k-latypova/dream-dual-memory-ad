import json
import torch
import os
from src.networks.resnet_anchor import ResNet_Model
from src.datasets.mvtec_pretrain_dataset import MvtecPretrainDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm

from src.utils.training_utils import cosine_annealing
from torch.utils.data.sampler import WeightedRandomSampler
from src.utils import constants

imagenet100_dirs = ['n01498041', 'n01514859', 'n01582220', 'n01608432', 'n01616318',
          'n01687978', 'n01776313', 'n01806567', 'n01833805', 'n01882714',
          'n01910747', 'n01944390', 'n01985128', 'n02007558', 'n02071294',
          'n02085620', 'n02114855', 'n02123045', 'n02128385', 'n02129165',
          'n02129604', 'n02165456', 'n02190166', 'n02219486', 'n02226429',
          'n02279972', 'n02317335', 'n02326432', 'n02342885', 'n02363005',
          'n02391049', 'n02395406', 'n02403003', 'n02422699', 'n02442845',
          'n02444819', 'n02480855', 'n02510455', 'n02640242', 'n02672831',
          'n02687172', 'n02701002', 'n02730930', 'n02769748', 'n02782093',
          'n02787622', 'n02793495', 'n02799071', 'n02802426', 'n02814860',
          'n02840245', 'n02906734', 'n02948072', 'n02980441', 'n02999410',
          'n03014705', 'n03028079', 'n03032252', 'n03125729', 'n03160309',
          'n03179701', 'n03220513', 'n03249569', 'n03291819', 'n03384352',
          'n03388043', 'n03450230', 'n03481172', 'n03594734', 'n03594945',
          'n03627232', 'n03642806', 'n03649909', 'n03661043', 'n03676483',
          'n03724870', 'n03733281', 'n03759954', 'n03761084', 'n03773504',
        #   'n03804744', 'n03916031', 'n03938244', 'n04004767', 'n04026417',
        #   'n04090263', 'n04133789', 'n04153751', 'n04296562', 'n04330267',
        #   'n04371774', 'n04404412', 'n04465501', 'n04485082', 'n04507155',
        #   'n04536866', 'n04579432', 'n04606251', 'n07714990', 'n07745940'
          ]

imagenet_classnames = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
                   'agama',   'tick', 'quail', 'hummingbird', 'koala',
                   'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
                   'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
                   'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
                   'monarch', 'starfish', 'hare', 'hamster', 'beaver',
                   'zebra', 'pig', 'ox', 'impala',  'mink',
                   'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
                   'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
                   'banjo','barn','baseball', 'basketball', 'beacon',
                   'binder', 'broom', 'candle', 'castle', 'chain',
                   'chest', 'church', 'cinema', 'cradle', 'dam',
                   'desk', 'dome', 'drum','envelope', 'forklift',
                   'fountain', 'gown', 'hammer','jean', 'jeep',
                   'knot', 'laptop', 'mower', 'library','lipstick',
                   'mask', 'maze', 'microphone','microwave','missile',
                #     'nail', 'perfume','pillow','printer','purse',
                #    'rifle', 'sandal', 'screw','stage','stove',
                #    'swing','television','tractor','tripod','umbrella',
                #     'violin','whistle','wreck', 'broccoli', 'strawberry'
                   ]


class MvtecADPreTrainer:
    def __init__(self, args: dict):
        self.args = args
        self.prompts = self.prepare_prompts(args)
        self.prepare_data(args)
        self.__load_embeds__()
        self.prepare_model(args)
        self.prepare_logs(args)

    def prepare_prompts(self, args):

        prompts = dict()
        for cat in constants.MVTEC_CATEGORIES:
            rootdir = self.args.mvtec_root if cat in constants.MVTEC_CATEGORIES_PER_DATASET["mvtec"] else self.args.mvtec2_root
            ds_path = os.path.join(rootdir, cat, "train", "good")
            if cat in constants.MVTEC_PROMPTS.keys():
                prompt = constants.MVTEC_PROMPTS[cat]
            else:
                prompt = cat
            prompts[ds_path] = prompt
        # prompts[mvtec_class_path] = args.prompt
        for imagenet_dir, imagenet_classname in zip(imagenet100_dirs, imagenet_classnames):
            imagenet_class_path = os.path.join(args.imagenet_root, imagenet_dir)
            if not os.path.exists(imagenet_class_path):
                continue
            prompts[imagenet_class_path] = imagenet_classname
        return prompts

    def __load_embeds__(self):
        with open(self.args.text_encoders_file, "r") as f:
            text_encoder_weights_mapping = json.load(f)
        anchor_path = os.path.join(self.args.output_dir, "anchor.npy")
        if os.path.exists(anchor_path):
            anchor = np.load(anchor_path)
            self.anchor = torch.from_numpy(anchor).to(self.args.device)
            print(f"Anchor was found in file: {anchor_path} and loaded")
            return
        self.model_name = "openai/clip-vit-large-patch14"  # Default model name, can be changed if needed
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        text_model = CLIPTextModel.from_pretrained(self.model_name).to(self.args.device)
        ds_embeddings = []
        for _, prompt in tqdm(self.prompts.items(), desc="Encoding prompts", leave=False):
            text_encoder_weights = text_encoder_weights_mapping.get(prompt)
            if text_encoder_weights:            
                weights = torch.load(text_encoder_weights, map_location=self.args.device)
                try:
                    text_model.load_state_dict(weights, strict=False)
                except Exception as e:
                    print(f"Failed to load weights for {prompt}. Error: {e}")
                    raise e
                text_model.eval()
            
            embedding_table = text_model.text_model.embeddings.token_embedding.weight
            tokenized = tokenizer(prompt, return_tensors="pt")
            for k, v in tokenized.items():
                tokenized[k] = v.to(self.args.device)
            #print(f"Class embeddings device: {next(embedding_table.parameters()).device}, token_ids device: {token_ids.device}")
            class_embeddings = embedding_table[tokenized['input_ids'][0][1]]
            ds_embeddings.append(class_embeddings)
        self.anchor = torch.stack(ds_embeddings).detach().to(self.args.device)
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        np.save(anchor_path, self.anchor.cpu().numpy())
        print(f"Finished preparing anchor embeddings with shape: {self.anchor.shape}. Saved to {anchor_path}")

        
    def prepare_data(self, args):
        self.dataset = MvtecPretrainDataset(prompts=self.prompts,
                                            transforms=T.Compose(
                                                [T.Resize((self.args.width, self.args.height)),
                                                 T.RandomHorizontalFlip(),
                                                 T.RandomCrop((args.crop_size, args.crop_size), padding=4),
                                                 T.ToTensor(),
                                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]
                                            ))
        assert len(np.unique(self.dataset.labels)) == len(self.prompts.keys())
        class_counts = np.bincount(self.dataset.labels)
        print(f"Class counts: {class_counts}")  # Debug: see your actual distribution
        num_classes = len(class_counts)
        class_weights = 1.0 / (class_counts.astype(float) + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize

        print(f"Class weights: {class_weights}")  # Debug

        # Create a weight for EACH SAMPLE based on its class
        sample_weights = np.array([class_weights[label] for label in self.dataset.labels])
        print(f"Sample weights shape: {sample_weights.shape}")  # Should be (total_samples,)

        # Now use the correct number of samples
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(self.dataset),  # Match dataset size!
            replacement=True
        )
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                     pin_memory=args.num_workers > 0,
                                     sampler = sampler,
                                     pin_memory_device=args.device, drop_last=True)
    
    def prepare_model(self, args):
        self.model = ResNet_Model(num_classes=len(self.prompts.keys())).to(args.device)
        if args.load_weights:
            init_weights = torch.load(args.load_weights, map_location=args.device)
            self.model.load_state_dict(init_weights)
            print(f"Model weights were loaded from {args.load_weights}")
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=(args.epochs - args.init_epoch) * len(self.dataloader),
            eta_min=1e-6
        )
        
    def prepare_logs(self, args):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        self.log_file = os.path.join(args.output_dir, "logs.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Training MVTec AD Pretrainer\n")
            f.write(f"Prompts: {self.prompts}\n")
            f.write(f"Anchor shape: {self.anchor.shape}\n")
        np.save(os.path.join(args.output_dir, "anchor.npy"), self.anchor.detach().cpu().numpy())
        print(f"Logs will be saved to {self.log_file}")
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(self.dataloader), leave=False, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            
            embeds = self.model(images)
            dists = F.cosine_similarity(self.anchor.reshape(1, -1, 768).repeat(len(embeds), 1, 1),
                                        embeds.unsqueeze(1).repeat(1, self.anchor.size(0), 1), dim=2)
            logits = dists /0.1
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            if batch_idx == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"\nEpoch {epoch}, Batch {batch_idx}: Gradient norm = {total_norm}")
                print(f"Loss = {loss.item()}")
                print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"Distance range: [{dists.min().item():.4f}, {dists.max().item():.4f}]")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            running_loss = running_loss * 0.8 + loss.item() * 0.2
            self.optimizer.step()
            self.scheduler.step()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': running_loss, 'acc': 100.0 * correct / total })
        accuracy = 100.0 * correct / total
        
        print(f"Epoch [{epoch + 1}/{self.args.epochs}], Loss: {running_loss}, Accuracy: {accuracy}")
        
        return running_loss
    
    def train(self):        
        for epoch in tqdm(range(self.args.init_epoch, self.args.epochs)):
            loss = self.train_epoch(epoch)
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch [{epoch + 1}/{self.args.epochs}], Loss: {loss / len(self.dataloader)}\n")
            if (epoch + 1) % 1 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, f"model_{epoch + 1}.pth"))
        
        # Save the model state      
        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "model.pth"))
        class_embeddings = self.get_class_embeddings()
        np.save(os.path.join(self.args.output_dir, "img_embeddings.npy"), class_embeddings)

    def get_class_embeddings(self) -> np.ndarray:
        """Get class embeddings from the model.

        Args:
            model (torch.nn.Module): Model to get embeddings from.
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (torch.device): Device to use for computation.

        Returns:
            torch.Tensor: Class embeddings.
        """
        self.model.eval()
        embeddings = []  # Assuming 768 is the embedding size
        our_ds_indices = [idx for idx, x in enumerate(self.dataset.labels) if x < len(constants.MVTEC_CATEGORIES)]
        ds = torch.utils.data.Subset(self.dataset, our_ds_indices)
        dataloader = DataLoader(ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    pin_memory=self.args.num_workers > 0, shuffle=False,
                                    pin_memory_device=self.args.device, drop_last=False)
        class_embeddings = [[] for _ in constants.MVTEC_CATEGORIES]
        with torch.no_grad():
            for data, labels in tqdm(dataloader):
                data = data.to(self.args.device)
                features = self.model(data).cpu()
                for i in range(data.shape[0]):
                    class_embeddings[labels[i]].append(features[i])
        embeddings = [np.stack(x).reshape(-1, 768) for x in class_embeddings]
        os.makedirs(os.path.join(self.args.output_dir, "embeds"), exist_ok=True)
        for cat, embed in zip(constants.MVTEC_CATEGORIES, embeddings):
            print(f"Embed shape: {embed.shape} for class {cat}")
            np.save(os.path.join(self.args.output_dir, "embeds", f"{cat}_embeddings"), embed)

        # return embeddings.reshape(len(constants.MVTEC_CATEGORIES), -1, 768).cpu().numpy()

           
        
        
