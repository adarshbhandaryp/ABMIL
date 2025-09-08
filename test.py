import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import cv2
from src.data_loader.mammo_transforms import BasicTransforms
import os
import torch
import wandb
from dataset import MammographyDataset, BreastMRI
#from src.models_all.models import create_soat_cnn_models
import torch.optim as optim
import pandas as pd
import argparse
import torch.nn as nn
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from torchinfo import summary
from src.trainer_tester.supervised_trainer import FullySupervisedMultiViewTrainer
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class BreastMRI(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir=None, split=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = np.stack([image] * 3, axis=-1)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pre_img = self._load_image(row['pre_png'])
        post_img = self._load_image(row['post_png'])
        sub_img = self._load_image(row['sub_png'])
        label = torch.tensor(row['Lesion']).long()
        patient_id = row['Patient_ID']
        return {
            'pre': pre_img,
            'post': post_img,
            'sub': sub_img,
            'label': label,
            'patient_id': patient_id
        }


class MultiViewSwinModel(nn.Module):
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



def has_valid_lesion(pred_classes, target_class):
        # Check for first or last slice being lesion
        if pred_classes[0] == target_class or pred_classes[-1] == target_class:
            return True
        # Check for 2 or more consecutive slices
        for i in range(len(pred_classes) - 1):
            if pred_classes[i] == target_class and pred_classes[i + 1] == target_class :
                return True
        return False



@torch.no_grad()
def evaluate(model, dataloader, device):
    read_ann = pd.read_csv("/anvme/workspace/b268dc11-breastt/BreastMRI/DataExtraction/annotated_odelia_slice_level.csv")
    read_ann = read_ann[read_ann["Split"].str.lower() == "test"]

    model.eval()
    patient_probs = defaultdict(list)
    patient_labels = {}

    for batch in dataloader:
        pre = batch['pre'].to(device)
        post = batch['post'].to(device)
        sub = batch['sub'].to(device)
        labels = batch['label']
        patient_ids = batch['patient_id']

        logits = model(pre, post, sub)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for i, pid in enumerate(patient_ids):
            patient_probs[pid].append(probs[i])
            patient_labels[pid] = labels[i].item()

    y_true = []
    y_pred = []
    y_true_onehot = []
    y_pred_onehot = []

    for pid in patient_probs:
        gt_label = patient_labels[pid]
        if pid in read_ann["UID"].values:
            rows = read_ann[read_ann["UID"] == pid]
            print("Tri-Slice Number(s):", rows["lesion_slice"].dropna().tolist())
            print("")
        else:
            print(f"{pid} not found in annotation.")
        slice_probs = np.array(patient_probs[pid])  # shape: (num_slices, num_classes)
        print("Patient ID:", pid, "GT label:", gt_label, "Slice probs shape:", np.round(slice_probs, 2))
        print("")
        predicted_classes = np.argmax(slice_probs, axis=1)
        print("")
        print("Predicted Classes:", predicted_classes)
        print("")
        # Check lesion logic
        has_malignant = has_valid_lesion(predicted_classes, target_class=2)
        has_benign = has_valid_lesion(predicted_classes, target_class=1)

        if has_malignant:
            idx = np.argmax(slice_probs[:, 2])  # slice with highest malignant prob
        elif has_benign:
            idx = np.argmax(slice_probs[:, 1])  # slice with highest benign prob
        else:
            idx = np.argmax(slice_probs[:, 0])  # slice with highest normal prob

        final_prob = slice_probs[idx]
        final_pred = np.argmax(final_prob)

        print(f"Evaluating patient: {pid} | GT label: {gt_label} | Pred: {final_pred} | MaxProbSlice: {idx}")
        print("")
        print("")
        print("")
        y_true.append(gt_label)
        y_pred.append(final_pred)
        y_true_onehot.append(label_binarize([gt_label], classes=[0, 1, 2])[0])
        y_pred_onehot.append(final_prob)

    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = np.vstack(y_true_onehot)
    y_pred_onehot = np.vstack(y_pred_onehot)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Normal', 'Benign', 'Malignant']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Patient-Level)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_patient_level.png")
    plt.close()

    # Metrics (only class labels used)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true_onehot, y_pred_onehot, average="micro")

    # ROC from binary class outputs
    fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_pred_onehot.ravel())
    sensitivity_at_90_spec = np.interp(0.10, fpr, tpr)
    specificity_at_90_sens = 1 - np.interp(0.90, tpr, fpr)
    avg_metric = np.mean([auc, sensitivity_at_90_spec, specificity_at_90_sens])

    # Results
    print("\n--- Evaluation Results (Per-Patient, Rule-Based Only) ---")
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"F1 Score (Macro)       : {f1:.4f}")
    print(f"AUC (Micro)            : {auc:.4f}")
    print(f"Sens @ 90% Specificity : {sensitivity_at_90_spec:.4f}")
    print(f"Spec @ 90% Sensitivity : {specificity_at_90_sens:.4f}")
    print(f"Average Metric         : {avg_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='swin_tiny_patch4_window7_224.ms_in22k')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BreastMRI(csv_file=args.csv_path, split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiViewSwinModel(model_name=args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    evaluate(model, dataloader, device)
