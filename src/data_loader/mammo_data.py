import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import pydicom
import einops

class Mammography(Dataset):

    def __init__(self, excel_file, data_folder, category, task, height, width, background_crop = True, use_clahe= False, transform=None):
        """
        Args:
            excel_file (string): Path to the excel file with annotations.
            category (string) : 'Classification' for Benign and Malignant. 'Subtypes' for Subtype Classification
            transform (callable, optional): Optional transform to be applied
        """
        self.mammography = pd.read_csv(excel_file, dtype = str)
        self.data_folder = data_folder
        self.category = category
        self.task = task
        self.height = height
        self.width = width
        self.background_crop  = background_crop
        self.use_clahe = use_clahe
        self.transform = transform

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, idx):

        if self.category == 'No Defects':
            class_abnormality = self.mammography.iloc[idx, 5]
            if class_abnormality in ['1', 1] : # No Defect   = True
                labels = 0.0
            elif class_abnormality in ['0', 0] : # No Defect   = False
                labels = 1.0
                
        if self.category == 'Skinfolds': ## Either 2i or 3i
            class_abnormality = self.mammography.iloc[idx, 6]
            if class_abnormality in ['0', 0 ] :
                labels = 0.0
            elif class_abnormality in ['1', 1]:
                labels = 1.0
                
        if self.category == '2i':
            class_abnormality = self.mammography.iloc[idx, 7]
            if class_abnormality in ['0', 0 ] :
                labels = 0.0
            elif class_abnormality in ['1', 1] :
                labels = 1.0
                
        if self.category == '3i':
            class_abnormality = self.mammography.iloc[idx, 8]
            if class_abnormality in ['0', 0 ] :
                labels = 0.0
            elif class_abnormality in ['1', 1] :
                labels = 1.0

        if self.category == 'Skinfold_MultiLabel':
            class_abnormality_skinfold = self.mammography.iloc[idx, 6]
            class_abnormality_defect = self.mammography.iloc[idx, 5]

            if class_abnormality_skinfold in ['0', 0 ] and  class_abnormality_defect in ['1', 1] :
                labels = [0., 0.]
            elif class_abnormality_skinfold in ['0', 0] and  class_abnormality_defect in ['0', 0 ]:
                labels = [0., 1.]
            elif class_abnormality_skinfold in ['1', 1] and  class_abnormality_defect in ['1', 1 ]:
                labels = [1., 0.]
            elif class_abnormality_skinfold in ['1', 1] and  class_abnormality_defect in ['0', 0 ]:
                labels = [1., 1.]

        if self.category == 'Skinfold_Defect_MultiLabel':
            class_abnormality_2i = self.mammography.iloc[idx, 7]
            class_abnormality_3i = self.mammography.iloc[idx, 8]
            class_abnormality_defect = self.mammography.iloc[idx, 5]

            if class_abnormality_2i in ['0', 0 ] and  class_abnormality_3i in ['0', 0 ]  and  class_abnormality_defect in ['1', 1 ]:
                labels = [0., 0., 0.]
            elif class_abnormality_2i in ['0', 0] and  class_abnormality_3i in ['1', 1 ] and  class_abnormality_defect in ['0', 0 ]:
                labels = [0., 1., 1.]
            elif class_abnormality_2i in ['1', 1] and  class_abnormality_3i in ['0', 0 ] and  class_abnormality_defect in ['0', 0 ]:
                labels = [1., 0., 1.]
            elif class_abnormality_2i in ['1', 1] and  class_abnormality_3i in ['1', 1 ] and  class_abnormality_defect in ['0', 0 ]:
                labels = [1., 1., 1.]

        if self.category == 'Calc_Mass_Malignant_MultiLabel':
            abnormality = self.mammography.iloc[idx, 3]
            classification = self.mammography.iloc[idx ,4]

            if abnormality == 'calcification' and classification == 'Benign':
                labels = [1. , 0., 0.]
            elif abnormality == 'both' and classification == 'Benign':
                labels = [1. , 1., 0.]
            elif abnormality == 'mass' and classification == 'Benign':
                labels = [0. , 1., 0.]
            elif abnormality == 'calcification' and classification == 'Malignant':
                labels = [1. , 0., 1.]
            elif abnormality == 'both' and classification == 'Malignant':
                labels = [1. , 1., 1.]
            elif abnormality == 'mass' and classification == 'Malignant':
                labels = [0. , 1., 1.]
            return labels

        return labels

    def detect_nonzero_regions(self, numbers):
        regions = []
        start = None
        
        for i, num in enumerate(numbers):
            if num != 0:
                if start is None:
                    start = i
            elif start is not None:
                regions.append((start, i-1))
                start = None
        
        if start is not None:
            regions.append((start, len(numbers)-1))
        return regions    
    
    def crop_images(self, data):
        columns = data.shape[1]
        count_all=[]
        for i in range(columns):
            count_non_zeros = np.count_nonzero(data[:, i][20:-20])
            count_all.append(count_non_zeros)
        column_indices = self.detect_nonzero_regions(count_all)
        differences = []
        for index, i in enumerate(column_indices):
            difference  = i[1]-i[0]
            differences.append(difference)

        index_max = np.argmax(differences)
        val = column_indices[index_max]
        if index_max == 0 and len(differences)!=1:
            column_indices = list(range(val[1], data.shape[1]))
        elif index_max == 0 and len(differences)==1:
            zero_columns = np.all(data == 0, axis=0)
            column_indices = np.where(zero_columns)[0]
        else:
            column_indices = list(range(0, val[0]))
        data = np.delete(data, column_indices, axis=1)
        zero_rows = np.all(data == 0, axis=1)
        row_indices = np.where(zero_rows)[0]
        data = np.delete(data, row_indices, axis=0)
        return data

    def image_load(self, idx, column):
        img_name = self.mammography.iloc[idx, column]
        
        if self.task == 'classification_dcm':
            image_name = os.path.join(self.data_folder, img_name +'.dcm')
            image = pydicom.dcmread(image_name)
            image_array = image.pixel_array
        elif self.task == 'classification_png':
            image_name = os.path.join(self.data_folder, img_name + '.png')
            image_array = cv2.imread(image_name)
        elif self.task == 'classification_jpeg':
            #filename = os.path.basename(img_name) ## Check why doesnt it work
            filename = img_name.split("\\")[-1] 
            image_name = os.path.join(self.data_folder, filename)
            image_array = cv2.imread(image_name)
            
        if self.task == 'classification_dcm' and self.background_crop:
            image_array = self.crop_images(image_array)
        image = cv2.resize(image_array, (self.width, self.height))

        
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) ## Using standard values; Extent of impact unknown.
            image = clahe.apply(image)
        image = image * 1.0 / image.max()
        image = torch.from_numpy(image)
        if self.task == 'classification_dcm':
            image = image[None, :, :]
            image = einops.repeat(image, 'b h w -> (repeat b) h w', repeat=3)
        elif self.task in ['classification_png', 'classification_jpeg']:
            image = image.permute(2, 0, 1)
        return image, image_name

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, image_name = self.image_load(idx, 9) 

        if self.transform:
            image = self.transform(image)
        labels = self.class_name_to_labels(idx)
        labels = torch.from_numpy(np.array(labels))
        
        if self.category in ['Calc_Mass_Malignant_MultiLabel']:
            ## For displaying the labels on the Output Image / GradCAM plots
            abnormality = self.mammography.iloc[idx, 3]
            classification = self.mammography.iloc[idx ,4]
            display_label = str(abnormality) + ' , ' + str(classification)

        else: 
            ## For displaying the labels on the Output Image / GradCAM plots
            two_i = self.mammography.iloc[idx, 7]
            three_i = self.mammography.iloc[idx, 8]
            defect = self.mammography.iloc[idx, 5]
            display_label = '2i:' + str(two_i)+ ' 3i:' + str(three_i) + ' Defect: ' + str(defect) 

        sample = {"image_name":image_name, "image": image, "label" : labels, "display_label":display_label}
        return sample