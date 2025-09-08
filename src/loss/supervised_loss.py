import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Segmentation Model Losses
Source: https://smp.readthedocs.io/en/latest/losses.html
'''


def set_classification_loss(weight, loss_function='CE'):
    if loss_function == 'MM':
        criterion = nn.MultiMarginLoss()
    elif loss_function == 'CE':
        criterion = nn.CrossEntropyLoss(weight=weight)

    return criterion


def set_segmentation_loss(loss_function='Dice',
                          mode='binary',
                          num_classes=None,
                          log_loss=False,
                          from_logits=True
                          ):
    if loss_function == 'Dice':
        criterion = smp.losses.DiceLoss(mode=mode,
                                        classes=None,
                                        log_loss=log_loss,
                                        from_logits=from_logits)

    elif loss_function == 'Jaccard':
        criterion = smp.losses.DiceLoss(mode=mode,
                                        classes=num_classes,
                                        log_loss=log_loss,
                                        from_logits=from_logits,
                                        smooth=0.0,
                                        eps=1e-07)

    return criterion


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input_embed, pos_embed, neg_embed):
        distance_pos = F.pairwise_distance(input_embed, pos_embed, p=2)
        distance_neg = F.pairwise_distance(input_embed, neg_embed, p=2)
        loss = torch.mean(torch.clamp(self.margin + distance_pos - distance_neg, min=0.0))
        return loss

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input_embed, pos_embed, neg_embed, weight=None):
        distance_pos = F.pairwise_distance(input_embed, pos_embed, p=2)
        distance_neg = F.pairwise_distance(input_embed, neg_embed, p=2)
        loss = torch.clamp(self.margin + distance_pos - distance_neg, min=0.0)
        if weight is not None:
            loss = (loss * weight / weight.sum()).sum()
        loss = torch.mean(loss)
        return loss
        
class WeightedCrossEntropy(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropy, self).__init__()

    def forward(self, weight=None):
        loss = nn.CrossEntropyLoss(reduction='none')
        if weight is not None:
            loss =(loss * weight / weight.sum()).sum()
        loss = loss.mean()
        #loss = torch.mean(loss)
        return loss