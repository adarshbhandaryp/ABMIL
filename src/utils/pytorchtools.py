import numpy as np
import torch


class EarlyStoppingLoss:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Source : https://github.com/Bjarten/early-stopping-pytorch

    """

    def __init__(self, patience=7,
                 verbose=False,
                 delta=0,
                 out_folder='.',
                 path='least_validation_loss.pth',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.out_folder = out_folder
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.out_folder + 'least_validation_model.pth')
        self.val_loss_min = val_loss


class EarlyStopping:
    """Early stops the training if validation AUC doesn't improve after a given patience."""

    def __init__(self, patience=7,
                 verbose=False,
                 delta=0.0,
                 out_folder='.',
                 path='best_auc_model.pth',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation AUC improved.
            verbose (bool): If True, prints a message for each validation AUC improvement.
            delta (float): Minimum change in AUC to qualify as improvement.
            out_folder (str): Folder to save checkpoint.
            path (str): Filename for saved model.
            trace_func (function): Custom print/logging function.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.out_folder = out_folder
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = float('-inf')

    def __call__(self, val_auc, model):
        """
        Call this at the end of each epoch.
        Args:
            val_auc (float): Validation AUC from current epoch.
            model (nn.Module): PyTorch model to be saved if AUC improves.
        """
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        """Save model when validation AUC improves."""
        if self.verbose:
            self.trace_func(
                f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.out_folder + '/' + self.path)
        self.val_auc_max = val_auc
