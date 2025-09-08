import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
import wandb
import time
import copy
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, method, model, criterion, optimizer, dataloaders, target_names, output_folders, num_epochs):
        self.method = method
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs

    def training_loop(self):
        train_loss_all = []
        self.optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(self.dataloaders['train']):
            self.optimizer.zero_grad()

            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            #print(inputs.shape)
            inputs, labels = inputs.cuda(), labels.long().cuda()
            bsz = labels.shape[0]
            features = self.model(inputs)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if self.method == 'SupCon':
                train_loss = self.criterion(features, labels)
            elif self.method == 'SimCLR':
                train_loss = self.criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(self.method))
            train_loss_all.append(train_loss.cpu().data.item())

            train_loss.backward()
            self.optimizer.step()
        return train_loss_all

    def validation_loop(self):
        validation_loss_all = []
        self.model.eval()
        with torch.no_grad():
            for inputs_v, labels_v in self.dataloaders['valid']:

                inputs_v = torch.cat([inputs_v[0], inputs_v[1]], dim=0)
                inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
                bsz_v = labels_v.shape[0]
                features_v = self.model(inputs_v)
                f1_v, f2_v = torch.split(features_v, [bsz_v, bsz_v], dim=0)
                features_v = torch.cat([f1_v.unsqueeze(1), f2_v.unsqueeze(1)], dim=1)
                if self.method == 'SupCon':
                    validation_loss = self.criterion(features_v, labels_v)
                elif self.method == 'SimCLR':
                    validation_loss = self.criterion(features_v)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                     format(self.method))
                validation_loss_all.append(validation_loss.cpu().data.item())
        return validation_loss_all

    def main_loop(self):
        training_losses = []
        for epoch in range(self.num_epochs):
            start = time.time()

            train_loss_all = self.training_loop()
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            training_losses.append(train_loss_avg)
            wandb.log({
                "Train Loss": train_loss_avg})
            end = time.time()
            print('Epoch: {}. '
                  'Time: {:.2f} '
                  'train_loss: {:.4f} '.format(epoch,
                                               end - start,
                                               train_loss_avg
                                               ))
            if train_loss_avg <= min(training_losses):
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, self.output_folders + 'least_training_loss_epoch.pth')
                del state
                wandb.run.summary["Least Train Loss"] = train_loss_avg

        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, self.output_folders + 'last_epoch.pth')
        return


class SupervisedTrainer:
    def __init__(self, device, method, model, classifier, criterion, optimizer, dataloaders, target_names,
                 output_folders, num_epochs):
        self.device = device
        self.method = method
        self.model = model
        self.classifier = classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.eval()
        self.classifier.train()

        for i, (inputs, labels) in enumerate(self.dataloaders['train']):
            inputs, labels = inputs.to(self.device), labels.long().to(self.device)
            self.optimizer.zero_grad()
            bsz = labels.shape[0]
            #labels =  labels.unsqueeze(1)
            with torch.no_grad():
                if self.method =='SupConEF':
                    features = self.model.calculate_embedding(inputs)
                elif self.method =='SupCon':
                    features = self.model.encoder(inputs)
            train_output = self.classifier(features.detach())
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
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for inputs_v, labels_v in self.dataloaders['valid']:
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.long().to(self.device)
                #labels_v =  labels_v.unsqueeze(1)
                bsz_v = labels_v.shape[0]
                if self.method =='SupConEF':
                    outputs_v = self.classifier(self.model.calculate_embedding(inputs_v))
                elif self.method =='SupCon':
                    outputs_v = self.classifier(self.model.encoder(inputs_v))
                
                
                
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())

                validation_loss_all.append(validation_loss.cpu().data.item())
            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])
            valid_f1 = f1_score(true, pred, average='macro')
            valid_accuracy = accuracy_score(true, pred)

            return valid_accuracy, valid_f1, true, pred, validation_loss_all

    def main_loop(self):
        validation_losses = []
        valids = []
        f1s = []
        for epoch in range(self.num_epochs):
            start = time.time()

            train_accuracy, train_f1, train_loss_all = self.training_loop()
            valid_accuracy, valid_f1, true, pred, validation_loss_all = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            validation_losses.append(validation_loss_avg)
            valids.append(valid_accuracy)
            f1s.append(valid_f1)

            wandb.log({
                "Train Loss": train_loss_avg,
                "Validation Loss": validation_loss_avg,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1})

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1:{:.4f}'.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1))
            # torch.save(self.model.state_dict(), self.output_folders + 'epoch_' + str(epoch) +'.pth')
            if validation_loss_avg <= min(validation_losses):
                torch.save(copy.deepcopy(self.model.state_dict()),
                           self.output_folders + 'least_validation_loss_epoch.pth')
                torch.save(copy.deepcopy(self.classifier.state_dict()),
                           self.output_folders + 'least_validation_loss_epoch_classifier.pth')

                print('Minimum Validation Loss')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Least Validation Loss"] = validation_loss_avg

            if valid_accuracy >= max(valids):
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_accuracy_epoch.pth')
                torch.save(copy.deepcopy(self.classifier.state_dict()),
                           self.output_folders + 'max_accuracy_epoch_classifier.pth')

                print('Maximum Accuracy')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Best Validation Accuracy"] = valid_accuracy

            if valid_f1 >= max(f1s):
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_f1_epoch.pth')
                torch.save(copy.deepcopy(self.classifier.state_dict()),
                           self.output_folders + 'max_f1_epoch_classifier.pth')
                print('Maximum F1')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Best Validation F1"] = valid_f1
            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        return


class FullySupervisedTrainer:
    def __init__(self,device, method, model, criterion, optimizer, dataloaders, target_names, output_folders,
                 num_epochs):
        self.device =device
        self.method = method
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.num_epochs = num_epochs

    def training_loop(self):
        train_loss_all = []
        labels_true_train = []
        labels_predicted_train = []
        self.model.train()

        for i, (inputs, labels) in enumerate(self.dataloaders['train']):
            #print(inputs.shape)
            #grid_img = torchvision.utils.make_grid(inputs, nrow=5).permute(1, 2, 0)
            #plt.imshow(grid_img)
            #plt.show()
            #plt.savefig('train_batch_'+str(i)+'.png')
            #labels = labels.type(torch.LongTensor) 
            #print('i:',1)
            inputs, labels = inputs.to(self.device), labels.long().to(self.device)
            
            self.optimizer.zero_grad()
            train_output = self.model(inputs)
            train_loss = self.criterion(train_output, labels)
            _, train_predicted = torch.max(train_output.data, 1)
            #train_predicted = np.round(train_output.cpu().detach())
            #labels = np.round(labels.cpu().detach())  
            
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
        self.model.eval()
        with torch.no_grad():
            #i=0
            for inputs_v, labels_v in self.dataloaders['valid']:
                #print(inputs.shape)
                #i+=1
                #grid_img = torchvision.utils.make_grid(inputs_v, nrow=5).permute(1, 2, 0)
                #plt.imshow(grid_img)
                #plt.savefig('valid_batch_'+str(i)+'.png')
                #labels_v = labels_v.type(torch.LongTensor) 
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.long().to(self.device)
                outputs_v = self.model(inputs_v)
                validation_loss = self.criterion(outputs_v, labels_v)
                _, predicted = torch.max(outputs_v.data, 1)
                #predicted = np.round(outputs_v.cpu().detach())
                #labels_v = np.round(labels_v.cpu().detach()) 
                
                labels_true.append(labels_v.tolist())
                labels_predicted.append(predicted.tolist())
                validation_loss_all.append(validation_loss.cpu().data.item())
            true = sum(labels_true, [])
            pred = sum(labels_predicted, [])
            valid_f1 = f1_score(true, pred, average='macro')
            valid_accuracy = accuracy_score(true, pred)

        return valid_accuracy, valid_f1, true, pred, validation_loss_all

    def main_loop(self):
        validation_losses = []
        valids = []
        f1s = []
        for epoch in range(self.num_epochs):
            start = time.time()

            train_accuracy, train_f1, train_loss_all = self.training_loop()
            valid_accuracy, valid_f1, true, pred, validation_loss_all = self.validation_loop()
            validation_loss_avg = (sum(validation_loss_all) / len(validation_loss_all))
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            validation_losses.append(validation_loss_avg)
            valids.append(valid_accuracy)
            f1s.append(valid_f1)

            wandb.log({
                "Train Loss": train_loss_avg,
                "Train Accuracy": train_accuracy,
                "Validation Loss": validation_loss_avg,
                "Validation Accuracy": valid_accuracy,
                "Validation F1": valid_f1})

            print('Epoch: {}. '
                  'train_loss: {:.4f} '
                  'train_accuracy: {:.4f} '
                  'validation_loss: {:.4f} '
                  'validation_accuracy: {:.4f} '
                  'valid_f1:{:.4f}'.format(epoch,
                                           train_loss_avg,
                                           train_accuracy,
                                           validation_loss_avg,
                                           valid_accuracy,
                                           valid_f1))
            # torch.save(self.model.state_dict(), self.output_folders + 'epoch_' + str(epoch) +'.pth')
            if validation_loss_avg <= min(validation_losses):
                torch.save(copy.deepcopy(self.model.state_dict()),
                           self.output_folders + 'least_validation_loss_epoch.pth')

                print('Minimum Validation Loss')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Least Validation Loss"] = validation_loss_avg

            if valid_accuracy >= max(valids):
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_accuracy_epoch.pth')

                print('Maximum Accuracy')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Best Validation Accuracy"] = valid_accuracy

            if valid_f1 >= max(f1s):
                torch.save(copy.deepcopy(self.model.state_dict()), self.output_folders + 'max_f1_epoch.pth')
                print('Maximum F1')
                print(classification_report(true, pred, target_names=self.target_names))
                wandb.run.summary["Best Validation F1"] = valid_f1
            end = time.time()
            print('Total time for One Epoch : {:.4f} Seconds ', end - start)
            print('')
        return
