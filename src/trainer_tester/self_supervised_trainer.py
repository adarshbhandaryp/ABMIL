import torch
import wandb
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, device, method, model, criterion, optimizer, dataloaders, target_names, output_folders, num_epochs):
        self.device = device
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
            inputs, labels = inputs.to(self.device), labels.long().to(self.device)
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
        return train_loss_all, features

    def validation_loop(self):
        validation_loss_all = []
        self.model.eval()
        with torch.no_grad():
            for inputs_v, labels_v in self.dataloaders['valid']:

                inputs_v = torch.cat([inputs_v[0], inputs_v[1]], dim=0)
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.long().to(self.device)
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
        return validation_loss_all, features_v

    def main_loop(self):
        training_losses = []
        valid_losses = []
        for epoch in range(self.num_epochs):
            start = time.time()

            train_loss_all, features_train = self.training_loop()
            valid_loss_all, features_v = self.validation_loop()
            #tsne = TSNE(n_components=3)
            #tsne_features = tsne.fit_transform(features_v.detach().cpu().numpy())
            #plt.scatter(tsne_features[:,0], tsne_features[:,1])
            #plt.savefig(self.output_folders + 'tsne'+str(epoch)+'.png')
            train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
            valid_loss_avg = (sum(valid_loss_all) / len(valid_loss_all))
            training_losses.append(train_loss_avg)
            valid_losses.append(valid_loss_avg)
            wandb.log({
                "Train Loss": train_loss_avg,
                "Validation Loss": valid_loss_avg})
            end = time.time()
            print('Epoch: {}. '
                  'Time: {:.2f} '
                  'train_loss: {:.4f} '
                  'valid_loss: {:.4f} '.format(epoch,
                                               end - start,
                                               train_loss_avg,
                                               valid_loss_avg
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
            
            if valid_loss_avg <= min(valid_losses):
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, self.output_folders + 'least_validation_loss_epoch.pth')
                del state
                wandb.run.summary["Least Valid Loss"] = valid_loss_avg
                
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, self.output_folders + 'last_epoch.pth')
        return