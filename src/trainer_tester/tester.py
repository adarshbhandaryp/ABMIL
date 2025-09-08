import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelBinarizer

class FullySupervisedTester:
    def __init__(self, model, dataloaders, target_names, output_folders):
        self.model = model
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders

    def test_loop(self):
        self.model.eval()
        with torch.no_grad():
            print('Predicting...')
            labels_true = []
            labels_predicted = []
            labels_total_one_hot = np.array([]).reshape((0, 6))
            outputs_preds = np.array([]).reshape((0, 6))
            for inputs_v, labels_v in self.dataloaders['test']:
                inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
                outputs_v = self.model(inputs_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                binarized = label_binarize(labels_v.cpu().numpy(), classes= [0,1,2,3,4,5])
                labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
            
        true = sum(labels_true, [])
        pred = sum(labels_predicted, [])
        valid_f1 = f1_score(true, pred, average='macro')
        print('F1', valid_f1)
        valid_accuracy = accuracy_score(true, pred)
        print('valid_accuracy', valid_accuracy)
        valid_roc_auc = roc_auc_score(labels_total_one_hot, outputs_preds, multi_class='ovr')
        return valid_accuracy, valid_f1, true, pred, labels_total_one_hot, outputs_preds, valid_roc_auc



class Tester:
    def __init__(self, model,classifier, dataloaders_test, target_names, output_folders, use_tta):
        self.model = model
        self.classifier = classifier
        self.dataloaders_test = dataloaders_test
        self.target_names = target_names
        self.output_folders = output_folders
        self.use_tta = use_tta

    def test_loop(self):
        
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            print('Predicting...')
            labels_true = []
            labels_predicted = []
            labels_total_one_hot = np.array([]).reshape((0, 6))
            outputs_preds = np.array([]).reshape((0, 6))
            for inputs_v, labels_v in self.dataloaders_test:
                inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
                if self.use_tta:
                    bs, ncrops, c, h, w = inputs_v.size()
                    outputs_v = self.model(inputs_v.view(-1, c, h, w))
                    outputs_v_avg = outputs_v.view(bs, ncrops, -1).mean(1)
                    _, predicted = torch.max(outputs_v_avg.data, 1)
                    # binarized = label_binarize(labels_v.cpu().numpy(), classes= [0,1])
                    # labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    # outputs_preds = np.concatenate((outputs_preds, outputs_v_avg.cpu().numpy()))
                    labels_true.append(labels_v.tolist())
                    labels_predicted.append(predicted.tolist())
                else:
                    outputs_v = self.classifier(self.model.encoder(inputs_v))
                    #print(outputs_v.shape)
                    _, predicted = torch.max(outputs_v.data, 1)
                    labels_true.append(labels_v.tolist())
                    labels_predicted.append(predicted.tolist())
                    # lb = LabelBinarizer()
                    # label = lb.fit_transform(labels_v.cpu().numpy())
                    # print(label.shape)
                    binarized = label_binarize(labels_v.cpu().numpy(), classes= [0,1,2,3,4,5])
                    #print(binarized.shape)
                    #print(labels_total_one_hot.shape)
                    #print(outputs_preds.shape)
                    labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
            
        true = sum(labels_true, [])
        pred = sum(labels_predicted, [])
        valid_f1 = f1_score(true, pred, average='macro')
        print('F1', valid_f1)
        valid_accuracy = accuracy_score(true, pred)
        print('valid_accuracy', valid_accuracy)
        valid_roc_auc = roc_auc_score(labels_total_one_hot, outputs_preds, multi_class='ovr')
        return valid_accuracy, valid_f1, true, pred , labels_total_one_hot, outputs_preds, valid_roc_auc


    def test_loop_two(self):
        
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            print('Predicting...')
            labels_true = []
            labels_predicted = []
            for inputs_v, labels_v in self.dataloaders_test:
                inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
                if self.use_tta:
                    bs, ncrops, c, h, w = inputs_v.size()
                    outputs_v = self.model(inputs_v.view(-1, c, h, w))
                    outputs_v_avg = outputs_v.view(bs, ncrops, -1).mean(1)
                    _, predicted = torch.max(outputs_v_avg.data, 1)
                    # binarized = label_binarize(labels_v.cpu().numpy(), classes= [0,1])
                    # labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    # outputs_preds = np.concatenate((outputs_preds, outputs_v_avg.cpu().numpy()))
                    labels_true.append(labels_v.tolist())
                    labels_predicted.append(predicted.tolist())
                else:
                    outputs_v = self.classifier(self.model.encoder(inputs_v))
                    print(outputs_v.shape)
                    _, predicted = torch.max(outputs_v.data, 1)
                    labels_true.append(labels_v.tolist())
                    labels_predicted.append(predicted.tolist())

                    #binarized = label_binarize(labels_v.cpu().numpy(), classes= [0,1,2,3,4,5])
                    #print(labels_total_one_hot.shape)
                    #print(outputs_preds.shape)
                    #labels_total_one_hot = np.concatenate((labels_total_one_hot, binarized))
                    #outputs_preds = np.concatenate((outputs_preds, outputs_v.cpu().numpy()))
            
        true = sum(labels_true, [])
        pred = sum(labels_predicted, [])
        valid_f1 = f1_score(true, pred, average='macro')
        print('F1', valid_f1)
        valid_accuracy = accuracy_score(true, pred)
        print('valid_accuracy', valid_accuracy)
        return valid_accuracy, valid_f1, true, pred #, labels_total_one_hot, outputs_preds
