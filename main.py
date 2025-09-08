import os
import torch
import wandb
from dataset import BreastMRI_ABMIL_NII, MammographyDataset, BreastMRI, BreastMRI_ABMIL
#from src.models_all.models import create_soat_cnn_models
import torch.optim as optim
import pandas as pd
import argparse
import torch.nn as nn
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from torchinfo import summary
from src.trainer_tester.supervised_trainer import FullySupervisedMultiViewTrainer, MultiInstanceTrainer
from src.utils.utils_functions import compute_sample_weights, log_scalar_values, set_seed
from src.data_loader.mammo_transforms import BasicTransforms, TrainTransform, TrainTransformBaseline
from src.utils.plotter import display_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.dataset_utils import extract_target_names
#import utils
import torch
import torch.nn as nn
import timm
from focal_loss.focal_loss import FocalLoss

class MultiViewSwinModel(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224.ms_in22k",
                 pretrained=True, num_classes=3, attn_dim=1024, num_heads=4):
        super(MultiViewSwinModel, self).__init__()

        # Load backbone without classification head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features  # should be 1024 for swin_base

        # Self-attention across views (3 views → sequence length 3)
        self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)

        # Final classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_feat(self, x):
        # Get unpooled features (B, H*W, C) and then pooled vector (B, C)
        features = self.backbone.forward_features(x)         # (B, H*W, C)
        pooled = self.backbone.forward_head(features, pre_logits=True)  # (B, C)
        return pooled

    def forward(self, img1, img2, img3):
        # Per-view embeddings
        f1 = self.extract_feat(img1)  # [B, C]
        f2 = self.extract_feat(img2)
        f3 = self.extract_feat(img3)

        # Stack views to form a sequence [B, 3, C]
        views = torch.stack([f1, f2, f3], dim=1)

        # Self-attention across views
        attn_out, _ = self.attn(views, views, views)  # [B, 3, C]

        # Aggregate (mean pooling across 3 views)
        fused = attn_out.mean(dim=1)  # [B, C]

        return self.classifier(fused)


class MultiViewSwinCrossAttn(nn.Module):
    """
    Cross‑attention Swin for 3‑view breast‑MRI (pre / post / subtraction).
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = True,
        num_classes: int = 3,
        num_xattn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Swin backbone WITHOUT final linear head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features        # 1024 for swin‑base

        # 2) Learnable CLS token (like ViT) and view‑type embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.view_embed = nn.Parameter(torch.zeros(3, 1, self.embed_dim))  # pre/post/sub

        # 3) Cross‑attention encoder (TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.xattn = nn.TransformerEncoder(encoder_layer, num_layers=num_xattn_layers)

        # 4) Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    # ------------------------------------------------------------------ #
    def _img_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward a single image through Swin and return pooled vector (B, C).
        """
        feats = self.backbone.forward_features(x)          # (B, H*W, C)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # (B, C)
        return pooled

    def forward(self, pre, post, sub):
        """
        Args:
            pre, post, sub: tensors (B, 3, 224, 224)
        Returns:
            logits (B, num_classes)
        """
        # 1) Per‑view embeddings
        z_pre = self._img_embed(pre)    # (B, C)
        z_post = self._img_embed(post)
        z_sub = self._img_embed(sub)

        # 2) Assemble sequence:  [CLS] + three view tokens
        B = z_pre.size(0)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, C)
        views = torch.stack([z_pre, z_post, z_sub], dim=1)  # (B, 3, C)

        # add view‑type embeddings (broadcast over batch)
        views = views + self.view_embed.transpose(0, 1)    # (B, 3, C)

        tokens = torch.cat([cls, views], dim=1)            # (B, 4, C)

        # 3) Cross‑attention
        tokens = self.xattn(tokens)                        # (B, 4, C)

        # 4) CLS pooling → head
        out = self.classifier(self.norm(tokens[:, 0]))     # (B, num_classes)
        return out

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm
import math
import numpy as np


class CrossModalAttentionABMIL_Swin(nn.Module):
    """
    Swin‑based Multiple‑Instance model with per‑slice cross‑modal attention
    over (pre, post, sub) features, followed by ABMIL pooling.

    Input  : (B, 32, 3, 224, 224)  # 3 modalities stacked in channel dim
    Output : logits (B, num_classes), attention‑per‑slice (B, 32)
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = True,
        num_classes: int = 3,
        hidden_dim: int = 256,
        cross_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # 1) Shared Swin backbone (no classifier head)
        # ------------------------------------------------------------
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 768 for swin_tiny

        # ------------------------------------------------------------
        # 2) Per‑slice cross‑modal fusion
        #    Input: three embeddings (pre, post, sub)  -> fused embedding
        # ------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cross_attn_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # One small transformer (2 layers) that attends across 3 tokens
        self.slice_fuser = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Token type embedding to differentiate modalities
        self.mod_embed = nn.Parameter(torch.randn(3, 1, self.embed_dim))

        # ------------------------------------------------------------
        # 3) ABMIL Attention across 32 fused slices
        # ------------------------------------------------------------
        self.attn_V = nn.Linear(self.embed_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, 1)

        # ------------------------------------------------------------
        # 4) Classifier head
        # ------------------------------------------------------------
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Init
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        nn.init.trunc_normal_(self.attn_V.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_U.weight, std=0.02)

    # ------------------------------------------------------------
    def _encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B*N, 1, 224, 224)  -> Swin pooled feats (B*N, embed_dim)
        """
        feats = self.backbone.forward_features(x)
        return self.backbone.forward_head(feats, pre_logits=True)

    # ------------------------------------------------------------
    def forward(self, volume: torch.Tensor):
        """
        volume : (B, 32, 3, 224, 224)
        """
        B, N, C_img, H, W = volume.shape        # C_img = 3 modalities
        assert C_img == 3, "Expect channel dim = 3 (pre, post, sub)"

        # --------------------------------------------------------
        # 1) Split modalities, flatten, and encode with Swin
        # --------------------------------------------------------
        # Each is (B*N, 1, H, W)
        def expand3(x): return x.expand(-1, 3, -1, -1)

        pre  = expand3(volume[:, :, 0, :, :].contiguous().view(B * N, 1, H, W))
        post = expand3(volume[:, :, 1, :, :].contiguous().view(B * N, 1, H, W))
        sub  = expand3(volume[:, :, 2, :, :].contiguous().view(B * N, 1, H, W))

        feat_pre  = self._encode_slices(pre).view(B, N, -1)   # (B, N, C)
        feat_post = self._encode_slices(post).view(B, N, -1)
        feat_sub  = self._encode_slices(sub).view(B, N, -1)

        # --------------------------------------------------------
        # 2) Cross‑modal attention fusion (per slice)
        # --------------------------------------------------------
        # Build token sequence [pre, post, sub] for each slice
        # Shape before fuser: (B, N, 3, C)  -> we fuse along dim=2
        slice_tokens = torch.stack([feat_pre, feat_post, feat_sub], dim=2)

        # Add modality embeddings
        slice_tokens = slice_tokens + self.mod_embed.transpose(0, 1)  # (B, N, 3, C)
        slice_tokens = slice_tokens.view(B * N, 3, self.embed_dim)    # (B*N, 3, C)

        # Transformer encoder attends across the 3 tokens
        fused = self.slice_fuser(slice_tokens)[:, 0]                  # take CLS‑like first token
        fused = fused.view(B, N, self.embed_dim)                      # (B, 32, C)

        # --------------------------------------------------------
        # 3) ABMIL attention over 32 fused slices
        # --------------------------------------------------------
        A = torch.tanh(self.attn_V(fused))                # (B, N, hidden)
        A = self.attn_U(A)                                # (B, N, 1)
        A = torch.softmax(A, dim=1)                       # (B, N, 1)
        patient_feat = (A * fused).sum(dim=1)             # (B, C)

        # --------------------------------------------------------
        # 4) Head
        # --------------------------------------------------------
        logits = self.classifier(self.dropout(self.norm(patient_feat)))
        return logits, A.squeeze(-1)                      # (B, num_classes), (B, 32)


class ABMIL_Swin(nn.Module):
    """
    Attention‑based Multiple‑Instance Learning (ABMIL) model
    for 3‑channel breast MRI slice triplets (pre / post / sub).

    Input shape : (B, 32, 3, 224, 224)
    Output      : logits (B, num_classes)  +  attention weights (B, 32)
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = True,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Swin backbone WITHOUT final linear head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # e.g. 768 (tiny), 1024 (base)

        # 2) Attention network (ABMIL)
        self.attn_V = nn.Linear(self.embed_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, 1)

        # 3) Patient‑level classifier
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Optional init
        nn.init.trunc_normal_(self.attn_V.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_U.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def _slice_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward one or more images through Swin and return pooled features.
        Args:
            x : (N, 3, 224, 224)
        Returns:
            pooled : (N, C)
        """
        feats = self.backbone.forward_features(x)                 # (N, H*W, C)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # (N, C)
        return pooled

    # ------------------------------------------------------------------ #
    def forward(self, volume: torch.Tensor):
        """
        Args:
            volume : (B, 32, 3, 224, 224)
        Returns:
            logits : (B, num_classes)
            attn_w : (B, 32)   -- attention per slice (sums to 1)
        """
        B, N, C, H, W = volume.shape
        x = volume.view(B * N, C, H, W)                # flatten slices
        
        # (B*N, 3, 224, 224) → (B*N, embed_dim) → reshape
        slice_feats = self._slice_embed(x).view(B, N, -1)  # (B, 32, embed_dim)

        # ----------------  ABMIL attention  ---------------- #
        A = torch.tanh(self.attn_V(slice_feats))            # (B, 32, hidden)
        A = self.attn_U(A)                                  # (B, 32, 1)
        A = torch.softmax(A, dim=1)                         # attention weights

        # Weighted sum → patient embedding
        patient_feat = (A * slice_feats).sum(dim=1)         # (B, embed_dim)

        # ----------------  Head  ---------------- #
        out = self.classifier(self.dropout(self.norm(patient_feat)))  # (B, num_classes)

        return out, A.squeeze(-1)  # logits, attention weights



def main(args):
    read_excel = pd.read_csv(args.configuration_file, skip_blank_lines=True, na_values=['NaN'])
    
    max_f1_scores = []
    model_files = []
    for index, row in read_excel.iterrows():
        if args.log_wandb:
            wandb.login()
            run = wandb.init(project=args.project_name, reinit=True, entity="adarshbhandary")

            with run:
                print('Experiment Index', index)
                torch.cuda.empty_cache()
                print("Reading Configuration File..")
                
                config = {
                        "cv_run": row['no'],
                        "task": row['task'],
                        "model": row['model'],
                        "method": row['method'],
                        "view": row['view'],
                        "pretrained": row['pretrained'],
                        "loss": row['loss'],
                        "optimizer": row['optimizer'],
                        "use_sampler": row['use_sampler'],
                        "height": int(row['height']),
                        "width": int(row['width']),
                        "background_crop": row['background_crop'],
                        "use_clahe": row['use_clahe'],
                        "batch_size": int(row['batch_size']),
                        "multi_gpu": bool(row['multi_gpu']),
                        "num_worker": row['num_worker'],
                        "learning_rate": row['lr'],
                        "weight_decay": row['weight_decay'],
                        "drop_out": row['drop_out'],
                        "num_epochs": int(row['epochs']),
                        "patience": row['patience'],
                        "probability": row['probability'],
                        "aug_mix_p":row['aug_mix_p'],
                        "erasing":row['erasing'],
                        "attention_head": row["attention_head"],
                        "rage":row["rage"],
                        "get_gradcam": bool(row['get_gradcam']),
                        "out_folder": row['out_folder'],
                        "folder_name": row['folder_name'],
                        "excel_file_train": row['train'],
                        "excel_file_validation": row['valid'],
                        "excel_file_test": row['test'],
                        "data_folder": row['data_folder'],
                        "pretrained_weights": row["pretrained_weights"],
                        "results_path": row['csv_results']
                    }
                set_seed(int(config["cv_run"]))
                wandb.config.update(config, allow_val_change=True)

                os.makedirs(os.path.join(config["out_folder"], str(config["folder_name"])), exist_ok=True)
                save_out_folder = config["out_folder"] + str(config["folder_name"])
                ## config method is MRI
                target_names = extract_target_names(method = config["method"])
                num_classes = len(target_names)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print('Device : ', device)
                
                weight, samples_weight = compute_sample_weights(config["excel_file_train"], class_type = config["method"], view = config["view"])
                if config["use_sampler"]:
                    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
                    shuffle = False
                else:
                    sampler = None
                    shuffle = True

                if config["model"] == "vit_small":
                    train_transforms = TrainTransform(aug=True, aug_mix_p = config["aug_mix_p"])
                else:
                    train_transforms = TrainTransformBaseline(aug=True, aug_mix_p = config["aug_mix_p"], erasing = config["erasing"])
                
                valid_transforms = BasicTransforms()

                if config["method"] == "MRI_MIL":
                    data_train =  BreastMRI_ABMIL_NII(
                                                    split='train'
                                                )

                    data_valid = BreastMRI_ABMIL_NII(
                                                split='val'
                                            )

                    data_test = BreastMRI_ABMIL_NII(
                                                split='test'
                                            )                                                                        

                else:
                    data_train = BreastMRI(csv_file  = config["excel_file_train"],  
                                                        root_dir=None,
                                                        split='train',
                                                        transform = valid_transforms)                               
                    data_valid = BreastMRI(csv_file  = config["excel_file_train"],  
                                                        root_dir=None,
                                                        split='val',
                                                        transform = valid_transforms)
                    data_test = BreastMRI(csv_file  = config["excel_file_train"],  
                                                        root_dir=None,
                                                        split='test',
                                                        transform = valid_transforms)

                print("Train dataset size:", len(data_train))
                print("Val dataset size:", len(data_valid))
                print("Test dataset size:", len(data_test))

                
                dataloader = {
                    'train': torch.utils.data.DataLoader(data_train,
                                                            batch_size=config["batch_size"],
                                                            shuffle=shuffle,
                                                            sampler=sampler,
                                                            num_workers=config["num_worker"],
                                                            pin_memory=True,
                                                            prefetch_factor=2,
                                                            drop_last=True
                                                            ),

                    'valid': torch.utils.data.DataLoader(data_valid,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            sampler=None,
                                                            num_workers=config["num_worker"],
                                                            pin_memory=True,
                                                            prefetch_factor=2,
                                                            drop_last=False
                                                            ),
                                                            
                    'test': torch.utils.data.DataLoader(data_test,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            sampler=None,
                                                            num_workers=config["num_worker"],
                                                            pin_memory=True,
                                                            prefetch_factor=2,
                                                            drop_last=False
                                                            )
                }
                
                if config["method"]=="MRI":
                    if config["model"] == "swin_cross":
                        model = MultiViewSwinCrossAttn(num_classes=num_classes)
                    else:
                        model = MultiViewSwinModel(num_classes=num_classes)
                elif config["method"] == "MRI_MIL":
                    if config["model"] == "swin_cross":
                        model = CrossModalAttentionABMIL_Swin(num_classes=num_classes)
                    else:
                        model = ABMIL_Swin(num_classes=num_classes)

                model = model.to(device)
                #summary(model, (config["batch_size"], 3, config["height"], config["width"]))
                if config["loss"] == "F":
                    criterion = FocalLoss(gamma=config["erasing"], reduction='mean').to(device)
                elif config["loss"] == "CE":
                    criterion = nn.CrossEntropyLoss().to(device)

                optimizer_ft = optim.Adam(model.parameters(),
                                            lr=config["learning_rate"],
                                            weight_decay=config["weight_decay"])

                if config["method"] == 'MRI':
                    train = FullySupervisedMultiViewTrainer(device,
                                                    config["method"],
                                                    model,
                                                    criterion,
                                                    optimizer_ft,
                                                    dataloader,
                                                    target_names,
                                                    save_out_folder,
                                                    config["num_epochs"],
                                                    config["patience"])
                    train.main_loop()
                    model.load_state_dict(torch.load(save_out_folder + 'max_metrics_epoch.pth'))

                    train = FullySupervisedMultiViewTrainer(device,
                                                    config["method"],
                                                    model,
                                                    criterion,
                                                    optimizer_ft,
                                                    dataloader,
                                                    target_names,
                                                    save_out_folder,
                                                    config["num_epochs"],
                                                    config["patience"])
                    test_accuracy, test_f1, true, pred, test_auc, averaged_results = train.test_loop()


                elif config["method"] == 'MRI_MIL':
                    #model.load_state_dict(torch.load(save_out_folder + 'max_metrics_epoch.pth'))
                    train = MultiInstanceTrainer(device,
                                                    config["method"],
                                                    model,
                                                    criterion,
                                                    config["loss"],
                                                    optimizer_ft,
                                                    dataloader,
                                                    target_names,
                                                    save_out_folder,
                                                    config["num_epochs"],
                                                    config["patience"])
                    train.main_loop()
                    model.load_state_dict(torch.load(save_out_folder + 'max_metrics_epoch.pth'))

                    train = MultiInstanceTrainer(device,
                                                    config["method"],
                                                    model,
                                                    criterion,
                                                    config["loss"],
                                                    optimizer_ft,
                                                    dataloader,
                                                    target_names,
                                                    save_out_folder,
                                                    config["num_epochs"],
                                                    config["patience"])
                    test_accuracy, test_f1, true, pred, test_auc, averaged_results = train.test_loop()





                print('Test Results')
                print(classification_report(true, pred, target_names=target_names, digits=4))
                display_confusion_matrix(true,
                                        pred,
                                        target_names,
                                        save_out_folder+'confusion_matrix.png',
                                        use_tta=False,
                                        label=config["task"],
                                        normalize=False)
                
                
                config["model location"] = save_out_folder + 'least_validation_loss.pth'
                config["test accuracy"] = test_accuracy
                config["test f1"] = test_f1
                config["test auc"] = test_auc
                
                #log_scalar_values(config["results_path"], config)
                max_f1_scores.append(test_f1)
                model_files.append(save_out_folder + 'least_validation_loss.pth')

                
                wandb.log({
                            "test_accuracy" : test_accuracy,
                            "test_f1" : test_f1,
                            "test_auc" : test_auc,     
                            "test_metrics" : averaged_results,                     
                            })
                
                #log_scalar_values(config["results_path"], config)
                wandb.config.update(config, allow_val_change=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration_file", type=Path, default='/cluster/panambur/DINO/config/transfer_baseline.csv',
                        help="""Change hyperparameters and input the filepath of CSV file""")
    parser.add_argument("--log_wandb", type=bool, default=True,
                        help="""Use FALSE for Siemens Code.""")
    parser.add_argument("--project_name", type=str, default='BreastMRI Tiny',
                        help="""For Wandb logging""")
    args = parser.parse_args()
    main(args)