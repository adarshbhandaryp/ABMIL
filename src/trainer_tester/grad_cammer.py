import torch
import os
import pandas as pd
from src.utils.grad_cam import grad_cam, save_grad_grad_cam
from src.utils.plotter import plot_grad_cam,plot_grad_cam_epoch,save_input_array_grad_array, plot_grad_cam_histogram, plot_grad_cam_single


class GRADCAMMER:
    def __init__(self, model, category, dataloaders_test, target_layer, output_folders):
        self.model = model
        self.category = category
        self.dataloaders_test = dataloaders_test
        self.output_folders = output_folders
        self.target_layer = target_layer
        #self.i = i

    def gradcam_loop(self):
        self.model.eval()

        count = 0
        for inputs_v, labels_v, file, abn in self.dataloaders_test:
            
            count += 1
            inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
            print('Preparing GRADCAM..')
            outputs_v = self.model(inputs_v)
            _, predicted1 = torch.max(outputs_v.data, 1)
            grad_image = grad_cam(self.model, inputs_v[0], self.target_layer, labels_v[0])
            plot_grad_cam(inputs_v, grad_image, count,  labels_v, predicted1, self.output_folders)
            #plot_grad_cam_histogram(inputs_v, grad_image, count,  labels_v, predicted1, self.output_folders)
        return

    def gradcam_get_masks(self):
        self.model.eval()
        count = 0
        filess= []
        imagessss = []
        maskssss  = []
        labels_true= []
        labels_predicted= []
        abns = []
        for inputs_v, labels_v, file, abn in self.dataloaders_test:
            count += 1
            inputs_v, labels_v = inputs_v.cuda(), labels_v.long().cuda()
            print('Preparing GRADCAM..', count)
            outputs_v = self.model(inputs_v)
            _, predicted1 = torch.max(outputs_v.data, 1)
            labels_true.append(labels_v.tolist())
            labels_predicted.append(predicted1.tolist())
            filess.append(file)
            name=str(count)
            newpath = os.path.join(self.output_folders,name) 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            input_image, grad_image = save_grad_grad_cam(self.model, inputs_v[0], self.target_layer, labels_v)
            image_filename, mask_filename = save_input_array_grad_array(inputs_v, grad_image, name, newpath)
            #plot_grad_cam(inputs_v, grad_image, count,  labels_v, predicted1, 'D:\\HRS_Anaylsis\\Experiments\\Resnet50\\Abnorormality\\valid\\gradimages\\')
            imagessss.append([image_filename])
            maskssss.append([mask_filename])
            abns.append(abn)
        true = sum(labels_true, [])
        pred = sum(labels_predicted, [])
        #files_final = sum(filess, [])
        imagessss_final = sum(imagessss, [])
        maskssss_final = sum(maskssss, [])
        print('true',len(true))
        print('pred',len(pred))
        print('filess',len(filess))
        print('imagessss_final',len(imagessss_final))
        print('maskssss_final',len(maskssss_final))
        df = pd.DataFrame(
                            {'Img_Folder': filess,
                             'True_Labels': true,
                             'Predicted_Labels': pred,
                             'Abnormality':abns,
                             'image_location':imagessss_final,
                             'mask_location':maskssss_final
                            })
        df.to_csv('pyradio_labels.csv')
        return true, pred
        