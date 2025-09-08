from torchvision import models
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
from vit_pytorch.crossformer import CrossFormer
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer
import timm

def create_soat_cnn_models(num_classes, model, pretrained, drop_out):
    if model == 'vgg19':
        model_ft = models.vgg19(pretrained=pretrained)
        number_features = model_ft.classifier[6].in_features
        if drop_out>0:
            model_ft.classifier[6] = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.classifier[6] = nn.Linear(number_features, num_classes)

    elif model == 'vgg19_bn':
        model_ft = models.vgg19_bn(pretrained=pretrained)
        number_features = model_ft.classifier[6].in_features
        if drop_out>0:
            model_ft.classifier[6] = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.classifier[6] = nn.Linear(number_features, num_classes)

    elif model == 'resnet18':
        model_ft = models.resnet18(pretrained=pretrained)
        number_features = model_ft.fc.in_features
        if drop_out>0:
            model_ft.fc = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.fc = nn.Linear(number_features, num_classes)

    elif model == 'resnet50':
        model_ft = models.resnet50(pretrained=pretrained)
        number_features = model_ft.fc.in_features
        if drop_out>0:
            model_ft.fc = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.fc = nn.Sequential(nn.Linear(number_features, num_classes))

    elif model == 'resnet152':
        model_ft = models.resnet152(pretrained=pretrained)
        number_features = model_ft.fc.in_features
        if drop_out>0:
            model_ft.fc = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.fc = nn.Linear(number_features, num_classes)

    elif model == 'densenet121':
        model_ft = models.densenet121(pretrained=pretrained)
        #print(model_ft)
        number_features = model_ft.classifier.in_features
        if drop_out>0:
            model_ft.classifier = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.classifier = nn.Linear(number_features, num_classes)

    elif model == 'densenet161':
        model_ft = models.densenet161(pretrained=pretrained)
        number_features = model_ft.classifier.in_features
        if drop_out>0:
            model_ft.classifier = nn.Sequential(nn.Dropout(drop_out), nn.Linear(number_features, num_classes))
        else:
            model_ft.classifier = nn.Linear(number_features, num_classes)

    elif model == 'skinfold_efficientnet':
        ## Dummy Implementation of a New Efficient Based Classifier; Half Image + Full Image + Half Image ; 
        ## Average performance; Maybe needs location of the Skinfold.
        model_ft = SkinFoldEfficientNet(num_classes=num_classes, drop_out=drop_out)
    
    
    elif model == 'efficientnet_b0':
        ## Author : Github : https://github.com/lukemelas/EfficientNet-PyTorch
        ## Repo with ImageNet Pretrained Models. 
        ## Some models in Torchvision are also ported from this repo; https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1
        model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, dropout_rate = drop_out)

    elif model == 'efficientnet_b3':
        ## Author : Github : https://github.com/lukemelas/EfficientNet-PyTorch
        ## Repo with ImageNet Pretrained Models. 
        ## Some models in Torchvision are also ported from this repo; https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1
        model_ft = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes, dropout_rate = drop_out)
        

    elif model == 'crossformer':
        ## Not Implemented
        model_ft = CrossFormer(
                            num_classes = num_classes,                # number of output classes
                            dim = (64, 128, 256, 512),         # dimension at each stage
                            depth = (2, 2, 8, 2),              # depth of transformer at each stage
                            global_window_size = (8, 4, 2, 1), # global window sizes at each stage
                            local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
                            dropout = drop_out,
                        )
        
    elif model =='max_vit':
        model_ft = timm.create_model('maxvit_base_tf_512.in21k_ft_in1k', 
                                     pretrained=pretrained, 
                                     pretrained_cfg_overlay=dict(file="/home/z003ve3e/MammographyAnalysis/external_model/maxvit_base_tf_512.in21k_ft_in1k/model.safetensors"),
                                     num_classes=num_classes, 
                                     drop_rate=drop_out
                                    )

    
    elif model == 'efficient_transformer':
        ## Not Implemented
        efficient_transformer = Nystromformer(
                                                dim = 512,
                                                depth = 12,
                                                heads = 8,
                                                num_landmarks = 256
                                            )

        model_ft = ViT(
                        dim = 512,
                        image_size = 1024,
                        patch_size = 32,
                        num_classes = num_classes,
                        transformer = efficient_transformer,
                        dropout = drop_out
                        )
            
    else:
        NotImplementedError("Model not implemented. Check src.models_all.models.py")

    return model_ft


class SkinFoldEfficientNet(nn.Module):

    ## Dummy Architecture; No performance gains. 

    def __init__(self, num_classes, drop_out):
        super(SkinFoldEfficientNet, self).__init__()

        # Main EfficientNet B0 for full image processing
        self.main_model = EfficientNet.from_pretrained('efficientnet-b0', dropout_rate = drop_out)
        self.main_model._fc = nn.Linear(self.main_model._fc.in_features, 512)

        # EfficientNet B0 for top half image processing
        self.top_model = EfficientNet.from_pretrained('efficientnet-b0', dropout_rate = drop_out)
        self.top_model._fc = nn.Linear(self.top_model._fc.in_features, 512)

        # EfficientNet B0 for bottom half image processing
        self.bottom_model = EfficientNet.from_pretrained('efficientnet-b0', dropout_rate = drop_out)
        self.bottom_model._fc = nn.Linear(self.bottom_model._fc.in_features, 512)

        self.fc = nn.Sequential(nn.Linear(1536, num_classes))

    def forward(self, x):
        # Split the input image into top and bottom halves
        top_half = x[:, :, :512, :]
        bottom_half = x[:, :, 512:, :]

        main_output = self.main_model(x)

        top_output = self.top_model(top_half)
        bottom_output = self.bottom_model(bottom_half)

        concatenated = torch.cat((main_output, top_output, bottom_output), dim=1)

        # Forward pass through the fully connected layer
        output = self.fc(concatenated)
        
        return output
