import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
import time
import copy
import wandb
from src.utils.pytorchtools import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path  
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize, LabelBinarizer
import os
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.tensorboard import SummaryWriter



class MultiInstanceTrainer:
    """
    Trainer for patient‑level Multiple‑Instance Learning (ABMIL) models.

    Expected dataloader output:
        data = {
            "volume": Tensor  # (B, 32, 3, 224, 224)
            "label" : Tensor  # (B,)
        }
    """

    def __init__(self, device, method, model, criterion, loss,  optimizer, dataloaders, target_names, output_folders,
                 num_epochs, patience):
        self.device = device
        self.method = method
        self.model = model
        self.loss = loss
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs
        self.patience = patience

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.train()
        for volumes, labels in self.dataloaders['train']:
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            train_output, _ = self.model(volumes)
            if self.loss == 'F':
                m = torch.nn.Softmax(dim=-1)
                train_output = m(train_output)
            train_loss = self.criterion(train_output, labels)
            _, train_predicted = torch.max(train_output.data, 1)
            labels_true_train.append(labels.tolist())
            labels_predicted_train.append(train_predicted.tolist())
            train_loss_all.append(train_loss.cpu().data.item())
            train_loss.backward()
            self.optimizer.step()

        true_train = sum(labels_true_train, [])
        pred_train = sum(labels_predicted_train, [])
        train_f1 = f1_score(true_train, pred_train, average='macro')
        train_accuracy = accuracy_score(true_train, pred_train)

        return train_accuracy, train_f1, train_loss_all

    def validation_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        labels_total_one_hot = np.array([]).reshape((0, 3))
        outputs_preds = np.array([]).reshape((0, 3))
        self.model.eval()
        with torch.no_grad():
            for volume_v, labels_v in self.dataloaders['valid']:
                volume_v = volume_v.to(self.device)
                labels_v = labels_v.to(self.device)
                outputs_v, _ = self.model(volume_v)
                if self.loss == 'F':
                    m = torch.nn.Softmax(dim=-1)
                    outputs_v = m(outputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())

                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))

            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])

            true_np = np.array(true)
            pred_np = np.array(pred)

            valid_f1 = f1_score(true_np, pred_np, average='macro')
            valid_accuracy = accuracy_score(true_np, pred_np)

            # Compute ROC and AUC for multiclass using one-vs-rest
            fpr, tpr, thresholds = roc_curve(labels_total_one_hot.ravel(), outputs_preds.ravel(), drop_intermediate=False)
            valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, average="micro")

            # Sensitivity at 90% specificity
            specificity_threshold = 0.90
            fpr_threshold = 1 - specificity_threshold
            sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

            # Specificity at 90% sensitivity
            sensitivity_threshold = 0.90
            fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
            specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

            amalgamated_results = [valid_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
            averaged_results = np.mean(amalgamated_results)

            return valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc, averaged_results

    def test_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        labels_total_one_hot = np.array([]).reshape((0, 3))
        outputs_preds = np.array([]).reshape((0, 3))
        self.model.eval()
        with torch.no_grad():
            for volume_v, labels_v in self.dataloaders['test']:
                volume_v = volume_v.to(self.device)
                print(volume_v.dtype)
                print(volume_v.shape)
                labels_v = labels_v.to(self.device)
                outputs_v, _ = self.model(volume_v)
                if self.loss == 'F':
                    m = torch.nn.Softmax(dim=-1)
                    outputs_v = m(outputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())

                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))

            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])

            true_np = np.array(true)
            pred_np = np.array(pred)

            valid_f1 = f1_score(true_np, pred_np, average='macro')
            valid_accuracy = accuracy_score(true_np, pred_np)

            # Compute ROC and AUC for multiclass using one-vs-rest
            fpr, tpr, thresholds = roc_curve(labels_total_one_hot.ravel(), outputs_preds.ravel(), drop_intermediate=False)
            valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, average="micro")

            # Sensitivity at 90% specificity
            specificity_threshold = 0.90
            fpr_threshold = 1 - specificity_threshold
            sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

            # Specificity at 90% sensitivity
            sensitivity_threshold = 0.90
            fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
            specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

            amalgamated_results = [valid_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
            averaged_results = np.mean(amalgamated_results)

            return valid_accuracy, valid_f1, true, pred, valid_auc, averaged_results
        

    def ensemble_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        labels_total_one_hot = np.array([]).reshape((0, 3))
        outputs_preds = np.array([]).reshape((0, 3))
        self.model.eval()
        with torch.no_grad():
            for volume_v, labels_v in self.dataloaders['test']:
                volume_v = volume_v.to(self.device)
                print(volume_v.dtype)
                print(volume_v.shape)
                labels_v = labels_v.to(self.device)
                outputs_v, _ = self.model(volume_v)
                if self.loss == 'F':
                    m = torch.nn.Softmax(dim=-1)
                    outputs_v = m(outputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())

                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))

            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])

            true_np = np.array(true)
            pred_np = np.array(pred)

            valid_f1 = f1_score(true_np, pred_np, average='macro')
            valid_accuracy = accuracy_score(true_np, pred_np)

            # Compute ROC and AUC for multiclass using one-vs-rest
            fpr, tpr, thresholds = roc_curve(labels_total_one_hot.ravel(), outputs_preds.ravel(), drop_intermediate=False)
            valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, average="micro")

            # Sensitivity at 90% specificity
            specificity_threshold = 0.90
            fpr_threshold = 1 - specificity_threshold
            sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

            # Specificity at 90% sensitivity
            sensitivity_threshold = 0.90
            fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
            specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

            amalgamated_results = [valid_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
            averaged_results = np.mean(amalgamated_results)

            return fpr, tpr , labels_total_one_hot, outputs_preds
    
    def main_loop(self):
        validation_losses = []
        valids = []
        f1s = []
        aucs = []
        metrics = []
        early_stopping = EarlyStopping(patience=self.patience,out_folder=self.output_folders,  verbose=True)
        writer = SummaryWriter(self.output_folders+'tensorboard_logs')
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()

            train_accuracy, train_f1, train_loss_all = self.training_loop()
            valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc, averaged_results  = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            early_stopping(averaged_results, self.model)
            validation_losses.append(validation_loss_avg)
            valids.append(valid_accuracy)
            f1s.append(valid_f1)
            aucs.append(valid_auc)
            metrics.append(averaged_results)

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1: {:.4f} '
                  'valid_metrics: {:.4f} '
                  'valid_auc: {:.4f} '.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1,
                                           averaged_results,
                                           valid_auc
                                           ))
            writer.add_scalar('Train Loss', train_loss_avg, global_step=epoch)
            writer.add_scalar('Train Accuracy', train_accuracy, global_step=epoch)
            writer.add_scalar('Validation Loss', validation_loss_avg, global_step=epoch)
            writer.add_scalar('Validation F1', valid_f1, global_step=epoch)
            writer.add_scalar('Validation AUC', valid_auc, global_step=epoch)
            writer.add_scalar('Validation Metrics', averaged_results, global_step=epoch)
            wandb.log({
                "Train Loss": train_loss_avg,
                "Train Accuracy": train_accuracy,
                "Validation Loss": validation_loss_avg,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1,
                "Validation Metrics": averaged_results,
                "Validation AUC": valid_auc})
                #"Validation AUC": valid_auc

            if averaged_results >= max(metrics):
                ## Not Required; Saving anyway
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_metrics_epoch.pth')
                print('Maximum AUC')
                print(classification_report(true, pred, target_names=self.target_names))
            
            if early_stopping.early_stop:
                ## Save on  the least validation loss
                print("Early stopping")
                break

            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        writer.close()
        return






class FullySupervisedMultiViewTrainer:

    def __init__(self, device, method, model, criterion, optimizer, dataloaders, target_names, output_folders,
                 num_epochs, patience):
        self.device = device
        self.method = method
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs
        self.patience = patience

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.train()
        for i, data in enumerate(self.dataloaders['train']):
            image_pre = data["pre"].to(self.device)
            image_post = data["post"].to(self.device)
            image_sub = data["sub"].to(self.device)
            labels = data["label"].to(self.device)
            self.optimizer.zero_grad()
            train_output = self.model(image_pre, image_post, image_sub)
            train_loss = self.criterion(train_output, labels)
            _, train_predicted = torch.max(train_output.data, 1)
            labels_true_train.append(labels.tolist())
            labels_predicted_train.append(train_predicted.tolist())
            train_loss_all.append(train_loss.cpu().data.item())
            train_loss.backward()
            self.optimizer.step()

        true_train = sum(labels_true_train, [])
        pred_train = sum(labels_predicted_train, [])
        train_f1 = f1_score(true_train, pred_train, average='macro')
        train_accuracy = accuracy_score(true_train, pred_train)

        return train_accuracy, train_f1, train_loss_all

    def validation_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        labels_total_one_hot = np.array([]).reshape((0, 3))
        outputs_preds = np.array([]).reshape((0, 3))
        self.model.eval()
        with torch.no_grad():
            for data in self.dataloaders['valid']:
                image_pre = data["pre"].to(self.device)
                image_post = data["post"].to(self.device)
                image_sub = data["sub"].to(self.device)
                labels_v = data["label"].to(self.device)
                outputs_v = self.model(image_pre, image_post, image_sub)
                
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())

                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))

            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])

            true_np = np.array(true)
            pred_np = np.array(pred)

            valid_f1 = f1_score(true_np, pred_np, average='macro')
            valid_accuracy = accuracy_score(true_np, pred_np)

            # Compute ROC and AUC for multiclass using one-vs-rest
            fpr, tpr, thresholds = roc_curve(labels_total_one_hot.ravel(), outputs_preds.ravel(), drop_intermediate=False)
            valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, average="micro")

            # Sensitivity at 90% specificity
            specificity_threshold = 0.90
            fpr_threshold = 1 - specificity_threshold
            sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

            # Specificity at 90% sensitivity
            sensitivity_threshold = 0.90
            fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
            specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

            amalgamated_results = [valid_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
            averaged_results = np.mean(amalgamated_results)

            return valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc, averaged_results

    def test_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        labels_total_one_hot = np.array([]).reshape((0, 3))
        outputs_preds = np.array([]).reshape((0, 3))
        self.model.eval()
        with torch.no_grad():
            for  data in self.dataloaders['test']:
                image_pre = data["pre"].to(self.device)
                image_post = data["post"].to(self.device)
                image_sub = data["sub"].to(self.device)
                labels_v = data["label"].to(self.device)
                outputs_v = self.model(image_pre, image_post, image_sub)
                
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())

                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))

            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])

            true_np = np.array(true)
            pred_np = np.array(pred)

            valid_f1 = f1_score(true_np, pred_np, average='macro')
            valid_accuracy = accuracy_score(true_np, pred_np)

            # Compute ROC and AUC for multiclass using one-vs-rest
            fpr, tpr, thresholds = roc_curve(labels_total_one_hot.ravel(), outputs_preds.ravel(), drop_intermediate=False)
            valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, average="micro")

            # Sensitivity at 90% specificity
            specificity_threshold = 0.90
            fpr_threshold = 1 - specificity_threshold
            sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

            # Specificity at 90% sensitivity
            sensitivity_threshold = 0.90
            fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
            specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

            amalgamated_results = [valid_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
            averaged_results = np.mean(amalgamated_results)

            return valid_accuracy, valid_f1, true, pred, valid_auc, averaged_results

    
    
    def main_loop(self):
        validation_losses = []
        valids = []
        f1s = []
        aucs = []
        metrics = []
        early_stopping = EarlyStopping(patience=self.patience,out_folder=self.output_folders,  verbose=True)
        writer = SummaryWriter(self.output_folders+'tensorboard_logs')
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()

            train_accuracy, train_f1, train_loss_all = self.training_loop()
            valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc, averaged_results  = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            early_stopping(averaged_results, self.model)
            validation_losses.append(validation_loss_avg)
            valids.append(valid_accuracy)
            f1s.append(valid_f1)
            aucs.append(valid_auc)
            metrics.append(averaged_results)

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1: {:.4f} '
                  'valid_metrics: {:.4f} '
                  'valid_auc: {:.4f} '.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1,
                                           averaged_results,
                                           valid_auc
                                           ))
            writer.add_scalar('Train Loss', train_loss_avg, global_step=epoch)
            writer.add_scalar('Train Accuracy', train_accuracy, global_step=epoch)
            writer.add_scalar('Validation Loss', validation_loss_avg, global_step=epoch)
            writer.add_scalar('Validation F1', valid_f1, global_step=epoch)
            writer.add_scalar('Validation AUC', valid_auc, global_step=epoch)
            writer.add_scalar('Validation Metrics', averaged_results, global_step=epoch)
            wandb.log({
                "Train Loss": train_loss_avg,
                "Train Accuracy": train_accuracy,
                "Validation Loss": validation_loss_avg,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1,
                "Validation Metrics": averaged_results,
                "Validation AUC": valid_auc})
                #"Validation AUC": valid_auc

            if averaged_results >= max(metrics):
                ## Not Required; Saving anyway
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_metrics_epoch.pth')
                print('Maximum AUC')
                print(classification_report(true, pred, target_names=self.target_names))
            
            if early_stopping.early_stop:
                ## Save on  the least validation loss
                print("Early stopping")
                break

            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        writer.close()
        return

class FullySupervisedTrainer:

    def __init__(self, device, method, model, linear_classifier, criterion, optimizer, dataloaders, target_names, output_folders,
                 num_epochs, patience):
        self.device = device
        self.method = method
        self.model = model
        self.linear_classifier = linear_classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs
        self.patience = patience

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.train()
        for i, (images, labels) in enumerate(self.dataloaders['train']):
            inputs = images.to(self.device)
            labels = labels.long().to(self.device)
            self.optimizer.zero_grad()
            intermediate_output = self.model.get_intermediate_layers(inputs, 4)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

            train_output = self.linear_classifier(output)
            train_loss = self.criterion(train_output, labels)
            _, train_predicted = torch.max(train_output.data, 1)
            labels_true_train.append(labels.tolist())
            labels_predicted_train.append(train_predicted.tolist())
            train_loss_all.append(train_loss.cpu().data.item())
            train_loss.backward()
            self.optimizer.step()

        true_train = sum(labels_true_train, [])
        pred_train = sum(labels_predicted_train, [])
        train_f1 = f1_score(true_train, pred_train, average='macro')
        train_accuracy = accuracy_score(true_train, pred_train)

        return train_accuracy, train_f1, train_loss_all

    def validation_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        if self.method == 'BreastDensity':
            labels_total_one_hot = np.array([]).reshape((0, 4))
            outputs_preds = np.array([]).reshape((0, 4))
        elif self.method in ['CM', 'Calc', 'Mass', 'CM_Malignant', 'MassNormal']:
            labels_total_one_hot = np.array([]).reshape((0, 3))
            outputs_preds = np.array([]).reshape((0, 3))
        elif self.method in ['CMMD_Malignant', 'MassRest']:
            labels_total_one_hot = np.array([]).reshape((0, 2))
            outputs_preds = np.array([]).reshape((0, 2))
        self.model.eval()
        self.linear_classifier.eval()
        with torch.no_grad():
            for inputs_valid, labels_valid in self.dataloaders['valid']:
                inputs_v = inputs_valid.to(self.device)
                labels_v = labels_valid.long().to(self.device)
                intermediate_output = self.model.get_intermediate_layers(inputs_v, 4)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

                outputs_v = self.linear_classifier(output)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())
                if self.method == 'BreastDensity':
                    binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2, 3])
                    labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
                elif self.method in ['CM', 'Calc', 'Mass', 'CM_Malignant','MassNormal']:
                    binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                    labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
                elif self.method in ['CMMD_Malignant', 'MassRest']:
                    binarized = 0
            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])
            valid_f1 = f1_score(true, pred, average='macro')
            valid_accuracy = accuracy_score(true, pred)
            if self.method not in ['CMMD_Malignant', 'MassRest']:
                valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, multi_class='ovr')
            else:
                valid_auc = roc_auc_score(true, pred)

        return valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc

    def test_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        if self.method == 'BreastDensity':
            labels_total_one_hot = np.array([]).reshape((0, 4))
            outputs_preds = np.array([]).reshape((0, 4))
        elif self.method in ['CM','CM_Malignant', 'Calc', 'Mass','MassNormal']:
            labels_total_one_hot = np.array([]).reshape((0, 3))
            outputs_preds = np.array([]).reshape((0, 3))
        elif self.method in ['CMMD_Malignant',  'MassRest']:
            labels_total_one_hot = np.array([]).reshape((0, 2))
            outputs_preds = np.array([]).reshape((0, 2))
        self.model.eval()
        self.linear_classifier.eval()
        with torch.no_grad():
            for inputs_valid, labels_valid in self.dataloaders['test']:
                inputs_v = inputs_valid.to(self.device)
                labels_v = labels_valid.long().to(self.device)
                intermediate_output = self.model.get_intermediate_layers(inputs_v, 4)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

                outputs_v = self.linear_classifier(output)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())
                if self.method == 'BreastDensity':
                    binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2, 3])
                    labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
                elif self.method in ['CM', 'Calc', 'Mass', 'CM_Malignant','MassNormal']:
                    binarized = label_binarize(labels_v.cpu().numpy(), classes= [0, 1, 2])
                    labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
                elif self.method in ['CMMD_Malignant', 'MassRest']:
                    binarized = 0
                
            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])
            valid_f1 = f1_score(true, pred, average='macro')
            valid_accuracy = accuracy_score(true, pred)
            if self.method not in ['CMMD_Malignant','MassRest']:
                valid_auc = roc_auc_score(labels_total_one_hot, outputs_preds, multi_class='ovr')
            else:
                valid_auc = roc_auc_score(true, pred)

        return valid_accuracy, valid_f1, true, pred, valid_auc
    
    
    def main_loop(self):
        validation_losses = []
        valids = []
        f1s = []
        aucs = []
        early_stopping = EarlyStopping(patience=self.patience,out_folder=self.output_folders,  verbose=True)
        writer = SummaryWriter(self.output_folders+'tensorboard_logs')
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()

            train_accuracy, train_f1, train_loss_all = self.training_loop()
            valid_accuracy, valid_f1, true, pred, validation_loss_all, valid_auc = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            early_stopping(validation_loss_avg, self.model, self.linear_classifier)
            validation_losses.append(validation_loss_avg)
            valids.append(valid_accuracy)
            f1s.append(valid_f1)
            aucs.append(valid_auc)

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1: {:.4f} '
                  'valid_auc: {:.4f} '.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1,
                                           valid_auc
                                           ))
            writer.add_scalar('Train Loss', train_loss_avg, global_step=epoch)
            writer.add_scalar('Train Accuracy', train_accuracy, global_step=epoch)
            writer.add_scalar('Validation Loss', validation_loss_avg, global_step=epoch)
            writer.add_scalar('Validation F1', valid_f1, global_step=epoch)
            writer.add_scalar('Validation AUC', valid_auc, global_step=epoch)

            wandb.log({
                "Train Loss": train_loss_avg,
                "Train Accuracy": train_accuracy,
                "Validation Loss": validation_loss_avg,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1,
                "Validation AUC": valid_auc})
                #"Validation AUC": valid_auc

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1:{:.4f}'
                  'valid_auc:{:.4f}'.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1,
                                           valid_auc
                                           ))


            if valid_f1 >= max(f1s):
                ## Not Required; Saving anyway
                #torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_f1_epoch.pth')
                print('Maximum F1')
                print(classification_report(true, pred, target_names=self.target_names))
            
            if valid_accuracy >= max(valids):
                ## Not Required; Saving anyway
                print('Maximum Accuracy')
                print(classification_report(true, pred, target_names=self.target_names))
            
            if valid_auc >= max(aucs):
                ## Not Required; Saving anyway
                #torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_auc_epoch.pth')
                print('Maximum AUC')
                print(classification_report(true, pred, target_names=self.target_names))
            
            if early_stopping.early_stop:
                ## Save on  the least validation loss
                print("Early stopping")
                break

            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        writer.close()
        return



class FullySupervisedTrainerMultiLabel:

    def __init__(self, device, method,  model, criterion, optimizer, dataloaders, target_names, output_folders,
                 num_epochs, patience, log_wandb):
        self.device = device
        self.method = method
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs
        self.patience = patience
        self.log_wandb = log_wandb 

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.train()
        for i, sample in enumerate(self.dataloaders['train']):
            inputs = sample['image'].to(self.device)
            labels = sample['label'].float().to(self.device)
            self.optimizer.zero_grad()
            train_output = self.model(inputs)
            train_loss = self.criterion(train_output, labels)
            labels_true_train.append(labels.tolist())
            labels_predicted_train.append(torch.sigmoid(train_output).tolist())
            train_loss_all.append(train_loss.cpu().data.item())
            train_loss.backward()
            self.optimizer.step()

        true_train = sum(labels_true_train, [])
        true_train = np.asarray(true_train)

        pred_train = sum(labels_predicted_train, [])
        pred_train = np.round(pred_train)


        if self.method in ['Calc_Mass_Malignant_MultiLabel']:
            train_accuracy_calcification = accuracy_score(true_train[:, 0], pred_train[:, 0])
            train_accuracy_mass = accuracy_score(true_train[:, 1], pred_train[:, 1])
            train_accuracy_malignancy = accuracy_score(true_train[:, 2], pred_train[:, 2])

            train_f1_calcification = f1_score(true_train[:, 0], pred_train[:, 0])
            train_f1_mass = f1_score(true_train[:, 1], pred_train[:, 1])
            train_f1_malignancy = f1_score(true_train[:, 2], pred_train[:, 2])
            
            return train_accuracy_calcification, train_accuracy_mass, train_accuracy_malignancy, train_f1_calcification, train_f1_mass, train_f1_malignancy, train_loss_all

        else:

            train_accuracy_2i = accuracy_score(true_train[:, 0], pred_train[:, 0])
            train_accuracy_3i = accuracy_score(true_train[:, 1], pred_train[:, 1])

            train_f1_2i = f1_score(true_train[:, 0], pred_train[:, 0])
            train_f1_3i = f1_score(true_train[:, 1], pred_train[:, 1])

            return train_accuracy_2i, train_accuracy_3i, train_f1_2i, train_f1_3i,  train_loss_all

    def validation_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in self.dataloaders['valid']:
                inputs_v = sample['image'].to(self.device)
                labels_v = sample['label'].float().to(self.device)
                outputs_v = self.model(inputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)

                labels_true.append(labels_v.tolist())
                labels_predicted.append(torch.sigmoid(outputs_v).tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())
                
            true_valid = sum(labels_true, [])
            true_valid = np.asarray(true_valid)

            pred_valid = sum(labels_predicted, [])
            pred_valid = np.round(pred_valid)

            if self.method in ['Calc_Mass_Malignant_MultiLabel']:
                valid_accuracy_calcification = accuracy_score(true_valid[:, 0], pred_valid[:, 0])
                valid_accuracy_mass = accuracy_score(true_valid[:, 1], pred_valid[:, 1])
                valid_accuracy_malignancy = accuracy_score(true_valid[:, 2], pred_valid[:, 2])


                valid_f1_calcification = f1_score(true_valid[:, 0], pred_valid[:, 0])
                valid_f1_mass = f1_score(true_valid[:, 1], pred_valid[:, 1])
                valid_f1_malignancy = f1_score(true_valid[:, 2], pred_valid[:, 2])


                valid_auc_calcification = roc_auc_score(true_valid[:, 0], pred_valid[:, 0])
                valid_auc_mass = roc_auc_score(true_valid[:, 1], pred_valid[:, 1])
                valid_auc_malignancy = roc_auc_score(true_valid[:, 2], pred_valid[:, 2])

                return valid_accuracy_calcification, valid_accuracy_mass, valid_accuracy_malignancy, valid_f1_calcification, valid_f1_mass, valid_f1_malignancy, valid_auc_calcification, valid_auc_mass, valid_auc_malignancy,  true_valid, pred_valid, validation_loss_all 
            
            else:

                valid_accuracy_2i = accuracy_score(true_valid[:, 0], pred_valid[:, 0])
                valid_accuracy_3i = accuracy_score(true_valid[:, 1], pred_valid[:, 1])

                valid_f1_2i = f1_score(true_valid[:, 0], pred_valid[:, 0])
                valid_f1_3i = f1_score(true_valid[:, 1], pred_valid[:, 1])

                valid_auc_2i = roc_auc_score(true_valid[:, 0], pred_valid[:, 0])
                valid_auc_3i = roc_auc_score(true_valid[:, 1], pred_valid[:, 1])

                return valid_accuracy_2i, valid_accuracy_3i, valid_f1_2i, valid_f1_3i, valid_auc_2i, valid_auc_3i,  true_valid, pred_valid, validation_loss_all 

    def test_loop(self):
        validation_loss_all = []
        labels_true = []
        labels_predicted = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in self.dataloaders['test']:
                inputs_v = sample['image'].to(self.device)
                labels_v = sample['label'].float().to(self.device)
                outputs_v = self.model(inputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(torch.sigmoid(outputs_v).tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())
                
            true_valid = sum(labels_true, [])
            true_valid = np.asarray(true_valid)

            pred_valid = sum(labels_predicted, [])
            pred_valid = np.round(pred_valid)

            if self.method in ['Calc_Mass_Malignant_MultiLabel']:
                valid_accuracy_calcification = accuracy_score(true_valid[:, 0], pred_valid[:, 0])
                valid_accuracy_mass = accuracy_score(true_valid[:, 1], pred_valid[:, 1])
                valid_accuracy_malignancy = accuracy_score(true_valid[:, 2], pred_valid[:, 2])


                valid_f1_calcification = f1_score(true_valid[:, 0], pred_valid[:, 0])
                valid_f1_mass = f1_score(true_valid[:, 1], pred_valid[:, 1])
                valid_f1_malignancy = f1_score(true_valid[:, 2], pred_valid[:, 2])


                valid_auc_calcification = roc_auc_score(true_valid[:, 0], pred_valid[:, 0])
                valid_auc_mass = roc_auc_score(true_valid[:, 1], pred_valid[:, 1])
                valid_auc_malignancy = roc_auc_score(true_valid[:, 2], pred_valid[:, 2])

                return valid_accuracy_calcification, valid_accuracy_mass, valid_accuracy_malignancy, valid_f1_calcification, valid_f1_mass, valid_f1_malignancy, valid_auc_calcification, valid_auc_mass, valid_auc_malignancy,  true_valid, pred_valid 
            
            else:

                valid_accuracy_2i = accuracy_score(true_valid[:, 0], pred_valid[:, 0])
                valid_accuracy_3i = accuracy_score(true_valid[:, 1], pred_valid[:, 1])

                valid_f1_2i = f1_score(true_valid[:, 0], pred_valid[:, 0])
                valid_f1_3i = f1_score(true_valid[:, 1], pred_valid[:, 1])

                valid_auc_2i = roc_auc_score(true_valid[:, 0], pred_valid[:, 0])
                valid_auc_3i = roc_auc_score(true_valid[:, 1], pred_valid[:, 1])

                return valid_accuracy_2i, valid_accuracy_3i, valid_f1_2i, valid_f1_3i, valid_auc_2i, valid_auc_3i,  true_valid, pred_valid 
    
    
    def main_loop_skinfold(self):
        validation_losses = []
        valid_accuracies_2i = []
        valid_accuracies_3i = []
        valid_f1s_2i = []
        valid_f1s_3i = []
        early_stopping = EarlyStopping(patience=self.patience,out_folder=self.output_folders,  verbose=True)
        if not self.log_wandb:
            writer = SummaryWriter(self.output_folders+'tensorboard_logs')
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            train_accuracy_2i, train_accuracy_3i, train_f1_2i, train_f1_3i,  train_loss_all = self.training_loop()
            valid_accuracy_2i, valid_accuracy_3i, valid_f1_2i, valid_f1_3i, valid_auc_2i, valid_auc_3i,   true_valid, pred_valid, validation_loss_all = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            early_stopping(validation_loss_avg, self.model)
            validation_losses.append(validation_loss_avg)
            valid_accuracies_2i.append(valid_accuracy_2i)
            valid_accuracies_3i.append(valid_accuracy_3i)
            valid_f1s_2i.append(valid_f1_2i)
            valid_f1s_3i.append(valid_f1_3i)
            print('Epoch: {} '
                  'train_loss: {:.4f} '
                  'train_accuracy_2i: {:.4f} '
                  'train_accuracy_3i: {:.4f} '
                  'valid_accuracy_2i: {:.4f} '
                  'valid_accuracy_3i: {:.4f} '
                  'valid_f1_2i: {:.4f} '
                  'valid_f1_3i: {:.4f} '
                  'valid_auc_2i: {:.4f} '
                  'valid_auc_3i: {:.4f} '
                  'validation_loss: {:.4f} '.format(epoch,
                                           train_loss_avg,
                                           train_accuracy_2i,
                                           train_accuracy_3i,
                                           valid_accuracy_2i,
                                           valid_accuracy_3i,
                                           valid_f1_2i,
                                           valid_f1_3i,
                                           valid_auc_2i,
                                           valid_auc_3i,
                                           validation_loss_avg))
            
            writer.add_scalar('Train Loss', train_loss_avg, global_step=epoch)
            writer.add_scalar('train_accuracy_2i', train_accuracy_2i, global_step=epoch)
            writer.add_scalar('train_accuracy_3i', train_accuracy_3i, global_step=epoch)
            writer.add_scalar('Validation Loss', validation_loss_avg, global_step=epoch)
            writer.add_scalar('valid_accuracy_2i', valid_accuracy_2i, global_step=epoch)
            writer.add_scalar('valid_accuracy_3i', valid_accuracy_3i, global_step=epoch)
            writer.add_scalar('valid_f1_2i', valid_f1_2i, global_step=epoch)
            writer.add_scalar('valid_f1_3i', valid_f1_3i, global_step=epoch)
            writer.add_scalar('valid_auc_2i', valid_auc_2i, global_step=epoch)
            writer.add_scalar('valid_auc_3i', valid_auc_3i, global_step=epoch)



            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        writer.close()
        return


    def main_loop_malignancy(self):
        validation_losses = []
        valid_accuracies_calcification = []
        valid_accuracies_mass = []
        valid_accuracies_malignancy = []
        valid_f1s_calcification = []
        valid_f1s_mass = []
        valid_f1s_malignancy = []
        early_stopping = EarlyStopping(patience=self.patience,out_folder=self.output_folders,  verbose=True)
        if not self.log_wandb:
            writer = SummaryWriter(self.output_folders+'tensorboard_logs')
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            train_accuracy_calcification, train_accuracy_mass, train_accuracy_malignancy, train_f1_calcification, train_f1_mass, train_f1_malignancy, train_loss_all = self.training_loop()
            valid_accuracy_calcification, valid_accuracy_mass, valid_accuracy_malignancy, valid_f1_calcification, valid_f1_mass, valid_f1_malignancy, valid_auc_calcification, valid_auc_mass, valid_auc_malignancy,  true_valid, pred_valid , validation_loss_all = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            early_stopping(validation_loss_avg, self.model)
            validation_losses.append(validation_loss_avg)
            
            valid_accuracies_calcification.append(valid_accuracy_calcification)
            valid_accuracies_mass.append(valid_accuracy_mass)
            valid_accuracies_malignancy.append(valid_accuracy_malignancy)


            valid_f1s_calcification.append(valid_f1_calcification)
            valid_f1s_mass.append(valid_f1_mass)
            valid_f1s_malignancy.append(valid_f1_malignancy)
            
            
            
            print('Epoch: {} '
                  'train_loss: {:.4f} '
                  'train_accuracy_calcification: {:.4f} '
                  'train_accuracy_mass: {:.4f} '
                  'train_accuracy_malignancy: {:.4f} '
                  'valid_accuracy_calcification: {:.4f} '
                  'valid_accuracy_mass: {:.4f} '
                  'valid_accuracy_malignancy: {:.4f} '
                  'valid_f1_calcification: {:.4f} '
                  'valid_f1_mass: {:.4f} '
                  'valid_f1_malignancy: {:.4f} '
                  'valid_auc_calcification: {:.4f} '
                  'valid_auc_mass: {:.4f} '
                  'valid_auc_malignancy: {:.4f} '
                  'validation_loss: {:.4f} '.format(epoch,
                                           train_loss_avg,
                                           train_accuracy_calcification,
                                           train_accuracy_mass,
                                           train_accuracy_malignancy,
                                           valid_accuracy_calcification,
                                           valid_accuracy_mass,
                                           valid_accuracy_malignancy,
                                           valid_f1_calcification,
                                           valid_f1_mass,
                                           valid_f1_malignancy,
                                           valid_auc_calcification,
                                           valid_auc_mass,
                                           valid_auc_malignancy, 
                                           validation_loss_avg))
            

            if not self.log_wandb:
                writer.add_scalar('Train Loss', train_loss_avg, global_step=epoch)
                writer.add_scalar('train_accuracy_calcification', train_accuracy_calcification, global_step=epoch)
                writer.add_scalar('train_accuracy_mass', train_accuracy_mass, global_step=epoch)
                writer.add_scalar('train_accuracy_malignancy', train_accuracy_malignancy, global_step=epoch)

                writer.add_scalar('Validation Loss', validation_loss_avg, global_step=epoch)
                writer.add_scalar('valid_accuracy_calcification', valid_accuracy_calcification, global_step=epoch)
                writer.add_scalar('valid_accuracy_mass', valid_accuracy_mass, global_step=epoch)
                writer.add_scalar('valid_accuracy_malignancy', valid_accuracy_malignancy, global_step=epoch)

                writer.add_scalar('valid_f1_calcification', valid_f1_calcification, global_step=epoch)
                writer.add_scalar('valid_f1_mass', valid_f1_mass, global_step=epoch)
                writer.add_scalar('valid_f1_malignancy', valid_f1_malignancy, global_step=epoch)
                
                writer.add_scalar('valid_auc_calcification', valid_auc_calcification, global_step=epoch)
                writer.add_scalar('valid_auc_mass', valid_auc_mass, global_step=epoch)
                writer.add_scalar('valid_auc_malignancy', valid_auc_malignancy, global_step=epoch)

            else:
                wandb.log({"Train Loss" : train_loss_avg,
                           "Train Accuracy Calcification" : train_accuracy_calcification,
                           "Train Accuracy Mass" : train_accuracy_mass,
                           "Train Accuracy Malignancy" : train_accuracy_malignancy,
                           "Valid Loss": validation_loss_avg,
                           "Valid Accuracy Calcification" : valid_accuracy_calcification,
                           "Valid Accuracy Mass" : valid_accuracy_mass,
                           "Valid Accurcay Malginancy" : valid_accuracy_malignancy,
                           "Valid F1 Calcification" : valid_f1_calcification,
                           "Valid F1 Mass" : valid_f1_mass,
                           "Valid F1 Malginancy" : valid_f1_malignancy,
                           "Valid AUC Calcification" : valid_auc_calcification,
                           "Valid AUC Mass" : valid_auc_mass,
                           "Valid AUC Malignancy" : valid_auc_malignancy
                           })

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        if not self.log_wandb:
            writer.close()
        return


class GetGradCam:
    def __init__(self, device, cam, model, dataloaders, target_names, output_folders):
        self.device = device
        self.cam = cam
        self.model = model
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders

    def gradcam(self):
        torch.set_grad_enabled(True)
        
        for sample in self.dataloaders['test']:
            inputs_v = sample['image'].to(self.device)
            labels_v = sample['label'].long().to(self.device)
            two_three = sample['two_three']
            outputs_v = self.model(inputs_v)
            _, predicted = torch.max(outputs_v.data, 1)
            predicted = predicted.tolist()
            filename = Path(sample['image_name'][0]).stem + '.png'
            targets = [ClassifierOutputTarget(labels_v)]
            grayscale_cam = self.cam(input_tensor=inputs_v, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            labels_v = labels_v.tolist()
            fig, axs = plt.subplots(1, 2)
            
            axs[0].imshow(inputs_v[0,:][0].cpu().numpy(), cmap = 'gray')
            axs[0].axis('off')
            axs[1].imshow(inputs_v[0,:][0].cpu().numpy(), cmap = 'gray')
            axs[1].imshow(grayscale_cam, alpha=0.5)  
            axs[1].axis('off')
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.1)
            fig.suptitle(str(two_three[0]) +' True :' + str(labels_v[0]) + ' Predicted :' + str(predicted[0]))
            os.makedirs(os.path.join(self.output_folders, 'gradcam'), exist_ok=True)
            plt.savefig(os.path.join(self.output_folders, 'gradcam', filename), bbox_inches = 'tight', dpi =300)
            plt.close()

        return 
    

class GetROIS:
    def __init__(self, device, cam, model, dataloaders, target_names, output_folders):
        self.device = device
        self.cam = cam
        self.model = model
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders

    def rois_from_gradcam(self):
        
        torch.set_grad_enabled(True)
        
        for sample in self.dataloaders:
            print('Generating ROIs..')
            inputs_v = sample['image'].to(self.device)
            labels_v = sample['label'].long().to(self.device)
            outputs_v = self.model(inputs_v)
            _, predicted = torch.max(outputs_v.data, 1)
            predicted = predicted.tolist()
            
            targets = [ClassifierOutputTarget(labels_v)]
            grayscale_cam = self.cam(input_tensor=inputs_v, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            labels_v = labels_v.tolist()
            threshold = 0.5  # Choose an appropriate threshold value
            binary_mask = (grayscale_cam > threshold).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            original_image = inputs_v[0,:][0].cpu().numpy()
            largest_contour = max(contours, key=cv2.contourArea)

            # Find the bounding box of the largest contour. Not Efficient, just a dummy implementation
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Extract the ROI from the original image
            roi_image = original_image[y:y+h, x:x+w]    
            os.makedirs(os.path.join(self.output_folders, 'rois_largest'), exist_ok=True)
            filename = Path(sample['image_name'][0]).stem +'.jpg'
            image_array = (roi_image * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_folders, 'rois_largest', filename), image_array, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return 