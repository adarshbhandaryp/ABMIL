import torch
import wandb
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import random
from math import sqrt

def dummy(lists):
    new = []
    for i in lists:
        new.append(i.item())
    return new  
    
def display(subject, ordered_t,count, num_images = 5):

    def l1_dist(x, y):
        return torch.sum(x - y).item()

    def l2_dist(x, y):
        
        return sqrt(torch.sum((x - y) ** 2).item())

    # sort by distance to the subject
    ordered_t = sorted(ordered_t, key=lambda elem: l2_dist(subject[0], elem[0]))

    subject_repeated = [subject for _ in range(num_images)]
    nearest_10_images = ordered_t[:num_images]
    farthest_10_images = ordered_t[-num_images:]

    def make_panel(list_of_images):
        images = [image[1] for image in list_of_images]
        labels = [image[2].cpu().numpy() for image in list_of_images]
        filenames = [image[3] for image in list_of_images]
        panel = torch.cat(images, dim=2)
        panel_pil = ToPILImage().__call__(panel)
        return panel_pil,labels, filenames

    panel_of_subject,labels_sub, filenames = make_panel(subject_repeated)
    #labels_sub = labels_sub.cpu().numpy()
    
    labels_sub = labels_sub[0]
    filename_sub = filenames[0]
    #print('Input:',labels_sub)
    panel_of_nearest_10,labels_near,_ = make_panel(nearest_10_images)
    labels_near = dummy(labels_near)
    #print('NN :',labels_near)
    panel_of_farthest_10,labels_far,_ = make_panel(farthest_10_images)

    #_img = np.concatenate((panel_of_subject, panel_of_nearest_10, panel_of_farthest_10), axis=0)
    #plt.title(str(labels_sub) + str(labels_near))
    #plt.imshow(_img)
    #plt.savefig('D:\\CalcSupContrastExperiments\\nn\\'+'image'+str(count) +'.png',dpi=200)
    #plt.close()
    return labels_sub, labels_near, filename_sub





class Nearest_Neigbours:
    def __init__(self, device, model, dataloaders, target_names, output_folders, name):
        self.device = device
        self.model = model
        self.dataloaders = dataloaders
        self.target_names = target_names
        self.output_folders = output_folders
        self.name = name

    def extract(self):
        self.model.eval()
        minibatches_t = []
        with torch.no_grad():
            for i, (inputs, labels, img_name) in enumerate(self.dataloaders):
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                
                features, feat = self.model(inputs)
                #features = features.view(features.size(0), -1)
                print(features.shape)
                i_t = inputs.detach().cpu().unbind(0)
                e_t = features.detach().cpu().unbind(0)
                sublist_t = [elem_t for elem_t in zip(e_t, i_t, labels, img_name)]
                minibatches_t.append(sublist_t)
        ordered_t = []
        for minibatch_t in minibatches_t:
            while minibatch_t:
                ordered_t.append(minibatch_t.pop())

        
        with open(self.output_folders + 'ordered_features_'+self.name , "wb") as fp:   #Pickling
            pickle.dump(ordered_t, fp)
            
        #with open(self.output_folders + 'labels_t'+self.name , "wb") as fp:   #Pickling
        #    pickle.dump(labels_t, fp)
            
            
        return ordered_t
   



    def main_loop(self):
        '''
        if os.path.exists(self.output_folders + 'ordered_features_train'):
            with open(self.output_folders + 'ordered_features_train', "rb") as fp:   # Unpickling
                ordered_t = pickle.load(fp)
            #with open(self.output_folders + 'labels_t'+self.name, "rb") as fp:   # Unpickling
            #    labels_t = pickle.load(fp)
            print('')
        else:
            ordered_t = self.extract()
        '''    
        if os.path.exists(self.output_folders + 'ordered_features_train_roi'):
            with open(self.output_folders + 'ordered_features_train_roi', "rb") as fp:   # Unpickling
                ordered_t_roi = pickle.load(fp)
            #with open(self.output_folders + 'labels_t'+self.name, "rb") as fp:   # Unpickling
            #    labels_t = pickle.load(fp)
            print('')
        else:
            ordered_t_roi = self.extract()
       
        #encoded_features = torch.stack(encoded_features, dim=0)
        num_features = len(ordered_t_roi)
        inputs=[]
        nns1 = []
        nns2 = []
        nns3 = []
        nns4 = []
        nns5 = []
        filess =[]
        encoded_representations = []
        count = 0
        for i in range(num_features):    
            count+=1
            # pick a random image
            print(count)
            subject = ordered_t_roi[i]
            
            
            labels_sub, labels_near, filename_sub = display(subject, ordered_t_roi, count, num_images = 5)
            extract_features = ordered_t_roi[i][0]
            #save_folder = os.makedirs(self.output_folders + 'extracted_features')
            pth_filename = os.path.splitext(os.path.basename(filename_sub))[0]
            pth_filename = self.output_folders + 'extracted_features/' + pth_filename +'.pth'
            torch.save(extract_features,pth_filename)
            filess.append(filename_sub)
            inputs.append(labels_sub)
            nns1.append(labels_near[0])
            nns2.append(labels_near[1])
            nns3.append(labels_near[2])
            nns4.append(labels_near[3])
            nns5.append(labels_near[4])
            encoded_representations.append(pth_filename)
        dataframe = pd.DataFrame({'Filename':filess,'Input':inputs,'NN1':nns1,'NN2':nns2,'NN3':nns3,'NN4':nns4,'NN5':nns5, 'Features':encoded_representations})
        dataframe.to_csv(self.output_folders+'New' +self.name+'_nn.csv', index = False)
        return