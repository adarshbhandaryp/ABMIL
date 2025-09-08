import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

class BreastMRI_ABMIL_NII(Dataset):

    def __init__(self, csv_file = "/anvme/workspace/b268dc11-breastt/BreastMRI/DataExtraction/combined_odelia_with_patient_split.csv", root_dir = "/anvme/workspace/b268dc11-breastt/OdeliaV2_extracted/OdeliaV2", split='train', transform=None):
        self.data = pd.read_csv(csv_file)
        print("Available splits:", self.data['Split_patient'].value_counts())

        if split is not None:
            self.data = self.data[self.data['Split_patient'] == split].reset_index(drop=True)
            print(f"Split = {split} -> {len(self.data)} samples")

        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)
    
    def process(self, dyn0, dyn1):
        sub = dyn1-dyn0
        sub = sub-sub.min() # Note: negative values causes overflow when using uint 
        sub = sub.astype(np.uint16)
        return sub
    
    def save_middle_slices_as_png(self, tensor, output_dir, prefix="sample"):
        
        os.makedirs(output_dir, exist_ok=True)

        channel_names = ['Pre', 'Post', 'Subtraction']

        # If batch dimension exists, take first sample
        if tensor.ndim == 4:
            tensor = tensor[0]  # shape (3, 224, 224)

        for i in range(3):
            slice_img = tensor[i].cpu().numpy()  # entire 2D image per channel
            plt.figure(figsize=(6,6))
            plt.imshow(slice_img, cmap='gray')
            plt.axis('off')
            filename = os.path.join(output_dir, f"{prefix}_{channel_names[i]}.png")
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    def resize(self, arr):
        # Initialize output array with target shape and same dtype
        arr_resized = np.zeros((arr.shape[0], 224, 224), dtype=arr.dtype)
        
        # Resize each slice individually
        for i in range(arr.shape[0]):
            arr_resized[i] = cv2.resize(arr[i], (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return arr_resized
    
    def normalize_per_channel_per_sample(self, x):
        # Compute min and max per sample and channel: shape (batch_size, channels, 1, 1)
        mins = x.amin(dim=[2,3], keepdim=True)
        maxs = x.amax(dim=[2,3], keepdim=True)
        
        # Avoid division by zero (if max == min)
        denom = (maxs - mins).clamp(min=1e-6)
        
        # Normalize
        x_norm = (x - mins) / denom
        return x_norm


    
    def _load_slices(self, row_value):
        filename = os.path.join(self.root_dir, row_value.replace('\\', '/'))
        
        # Try posts in descending order
        post_files = ["Post_4.nii.gz", "Post_3.nii.gz", "Post_2.nii.gz"]
        post_path = None
        
        for pf in post_files:
            candidate = os.path.join(filename, pf)
            if os.path.exists(candidate):
                post_path = candidate
                break
        
        if post_path is None:
            raise FileNotFoundError(f"No Post_* file found in {filename}")
        
        pre_path = os.path.join(filename, "Pre.nii.gz")
        post_1_path = os.path.join(filename, "Post_1.nii.gz")
        pre_sequence = sitk.ReadImage(pre_path)
        post_sequence = sitk.ReadImage(post_path)
        post_1_sequence = sitk.ReadImage(post_1_path)
        
        pre_array = sitk.GetArrayFromImage(pre_sequence)
        post_1_array = sitk.GetArrayFromImage(post_1_sequence)
        post_array = sitk.GetArrayFromImage(post_sequence)
        sub_array = self.process(pre_array, post_1_array)

        pre_array = self.resize(pre_array)
        post_array = self.resize(post_array)
        sub_array = self.resize(sub_array)

        stacked_array = np.stack([pre_array, post_array, sub_array], axis=1)
        stacked_tensor = torch.from_numpy(stacked_array).float()
        normalized_tensor  = self.normalize_per_channel_per_sample(stacked_tensor)

        return normalized_tensor




    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"[{self.__class__.__name__}] Index {idx} is out of bounds for dataset of length {len(self.data)}")

        row = self.data.iloc[idx]
        path = row['SamplePath']
        stacked_tensor = self._load_slices(path)
        #if idx<3:
        #    self.save_middle_slices_as_png(stacked_tensor, output_dir="./visualizations", prefix=f"sample_{idx}")
        label = torch.tensor(row['Lesion']).long()
        return stacked_tensor, label


class BreastMRI_ABMIL(Dataset):
    """
    PyTorch Dataset for patient-level ABMIL training using 32 slices per patient.

    Expects a CSV with columns:
    - 'UID' (unique per patient)
    - 'SamplePath' (relative path from root_dir)
    - 'Lesion' (patient-level label)
    
    Expects two root folders:
    - One for pre/post slices: root_dir_prepost
    - One for sub slices: root_dir_sub

    For each UID, constructs 32 slices:
    - Each slice is a 3-channel image: [pre, post, sub]
    - Returns a tuple: (tensor of shape [32, 3, 224, 224], label)
    """

    def __init__(self, csv_file, root_dir_prepost, root_dir_sub, split='train', transform=None):
        self.data = pd.read_csv(csv_file)
        print("Available splits:", self.data['Split'].value_counts())

        if split is not None:
            self.data = self.data[self.data['Split'] == split].reset_index(drop=True)
            print(f"Split = {split} -> {len(self.data)} samples")

        self.root_dir_prepost = root_dir_prepost
        self.root_dir_sub = root_dir_sub
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_slice_triplet(self, pre_path, post_path, sub_path):
        def read_img(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            return torch.from_numpy(img).float() / 255.0  # (H, W)

        pre = read_img(pre_path)
        post = read_img(post_path)
        sub = read_img(sub_path)
        stacked = torch.stack([pre, post, sub], dim=0)  # (3, H, W)

        if self.transform:
            stacked = self.transform(stacked)

        return stacked  # (3, H, W)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"[{self.__class__.__name__}] Index {idx} is out of bounds for dataset of length {len(self.data)}")

        row = self.data.iloc[idx]
        uid = row['UID']
        sample_path = row['SamplePath'].replace("\\", "/")  # normalize

        prepost_dir = os.path.join(self.root_dir_prepost, sample_path)
        sub_dir = os.path.join(self.root_dir_sub, sample_path)

        slice_stack = []
        for i in range(32):
            slice_id = f"{i:03d}"
            pre_path = os.path.join(prepost_dir, f"{uid}_pre_slice_{slice_id}.png")
            post_path = os.path.join(prepost_dir, f"{uid}_post_slice_{slice_id}.png")
            sub_path = os.path.join(sub_dir, f"{uid}_sub_slice_{slice_id}.png")

            try:
                slice_img = self._load_slice_triplet(pre_path, post_path, sub_path)
            except Exception as e:
                raise FileNotFoundError(f"Error loading slice {i} for UID {uid}: {e}")

            slice_stack.append(slice_img)

        volume = torch.stack(slice_stack, dim=0)  # (32, 3, 224, 224)
        label = torch.tensor(row['Lesion']).long()

        return volume, label




class BreastMRI(Dataset):
    """
    PyTorch Dataset for breast MRI slices with pre-contrast, post-contrast, and subtraction images.

    Expects a CSV with columns 'Split', 'pre_png', 'post_png', 'sub_png', and 'Lesion'.
    Paths in the CSV may be absolute or relative to `root_dir`.

    Returns a dict with keys 'pre', 'post', 'sub', and 'label'.
    """

    def __init__(self, csv_file, root_dir=None, split='train', transform=None):
        # Load the annotations CSV
        self.data = pd.read_csv(csv_file)
        print("Available splits:", self.data['Split'].value_counts())

        if split is not None:
            self.data = self.data[self.data['Split'] == split].reset_index(drop=True)
            print(f"Split = {split} -> {len(self.data)} samples")
        # Filter by split if provided

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_image(self, img_path):

        # Read as grayscale (2D)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224,224))
        image = np.stack([image] * 3, axis=-1)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Optionally apply transform
        if self.transform:
            # Assume transform can handle single-channel images
            image = self.transform(image)
        else:
            # Convert to tensor with shape (1, H, W)
            image = torch.from_numpy(image).unsqueeze(0).float()
        return image

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"[{self.__class__.__name__}] Index {idx} is out of bounds for dataset of length {len(self.data)}")

        row = self.data.iloc[idx]

        try:
            pre_img = self._load_image(row['pre_png'])
            post_img = self._load_image(row['post_png'])
            sub_img = self._load_image(row['sub_png'])
        except Exception as e:
            raise FileNotFoundError(f"Could not load one of the images: {e}\nRow:\n{row}")

        label = torch.tensor(row['Lesion']).long()

        return {
            'pre': pre_img,
            'post': post_img,
            'sub': sub_img,
            'label': label
        }



class MammographyDataset(Dataset):

    def __init__(self, split_file, data_folder, task, view, probability, split, attention_head, transform=None, rage = True):

        self.mammography = pd.read_csv(split_file) 
        if view in ['CC', 'MLO']: 
            self.mammography = self.mammography[self.mammography['View Position'] == view]
        self.data_folder = data_folder
        self.task = task
        self.transform = transform
        self.probability = probability
        self.split = split
        self.attention_head = attention_head
        self.rage = rage 

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, idx):
        if self.task == 'BreastDensity':
            class_name = self.mammography.iloc[idx, 8]
            if class_name in ['DENSITY A']:
                labels = 0.0
            elif class_name in ['DENSITY B']:
                labels = 1.0
            elif class_name in ['DENSITY C']:
                labels = 2.0
            elif class_name in ['DENSITY D']:
                labels = 3.0
            return labels

        elif self.task == 'BIRADS':
            class_name = self.mammography.iloc[idx, 7]
            if class_name in ['BI-RADS 1']:
                labels = 0.0
            elif class_name in ['BI-RADS 2']:
                labels = 1.0
            elif class_name in ['BI-RADS 3']:
                labels = 2.0
            elif class_name in ['BI-RADS 4']:
                labels = 3.0
            elif class_name in ['BI-RADS 5']:
                labels = 4.0
            return labels
            
        elif self.task == 'MassRest':
            class_name = self.mammography.iloc[idx, 30]
            if 'Mass' in class_name:
                labels = 0.0
            else:
                labels = 1.0
            return labels
            
        elif self.task == 'MassNormal':
            class_name = self.mammography.iloc[idx, 30]
            if 'Mass' in class_name:
                labels = 0.0
            elif 'No Finding' in class_name:
                labels = 1.0
            else:
                labels = 2.0
            return labels
        
        
        elif self.task == 'CMMD_Malignant':
            class_name = self.mammography.iloc[idx, 4]
            if class_name in ['Benign']:
                labels = 0.0
            elif class_name in ['Malignant']:
                labels = 1.0
            return labels
        
        elif self.task == 'CM':
            class_name = self.mammography.iloc[idx, 11]
            if class_name in ['Normal']:
                labels = 0.0
            elif class_name in ['Benign']:
                labels = 1.0
            elif class_name in ['Malignant']:
                labels = 2.0
            return labels
        
        elif self.task == 'CM_Malignant':
            class_name = self.mammography.iloc[idx, 11]
            if class_name in ['Normal']:
                labels = 0.0
            elif class_name in ['Benign']:
                labels = 1.0
            elif class_name in ['Malignant']:
                labels = 2.0
            return labels
        
        elif self.task in ['Calc', 'Mass']:
            class_name = self.mammography.iloc[idx, 9]
            if class_name in ['BENIGN_WITHOUT_CALLBACK']:
                labels = 0.0
            elif class_name in ['BENIGN']:
                labels = 1.0
            elif class_name in ['MALIGNANT']:
                labels = 2.0
            return labels

    def image_load(self, idx, img_column, folder_column):

        if self.task in ['BreastDensity', 'MassRest', 'MassNormal']:

            img_name = self.mammography.iloc[idx, img_column]
            img_folder = self.mammography.iloc[idx, folder_column]
            image_name = os.path.join(self.data_folder, img_folder, img_name, "img.png")
        
        elif self.task in ['CM', 'Calc', 'Mass', 'CMMD_Malignant', 'CM_Malignant']:
            img_filename = self.mammography.iloc[idx, img_column]
            img_folder = img_filename.split("/")[-1][:-4]
            image_name = os.path.join(self.data_folder, img_folder, "img.png")
            #print("Image Name", image_name)
        #image_name = image_name + '/img.png'
        image = cv2.imread(image_name)
        image[image < 40] = 0 # had it for breast density, why ? 
        image = cv2.resize(image, (448,448))
        return image
    
    def convert_to_mask(self, image, threshold):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_mask

    def mask_load(self, idx, img_column, folder_column):

        if self.task in ['BreastDensity', 'MassRest', 'MassNormal']:
            img_name = self.mammography.iloc[idx, img_column]
            img_folder = self.mammography.iloc[idx, folder_column]
            if self.rage:
                self.attention_head=random.randint(0,5)
            if self.attention_head in ['0', 0]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head0.png")
            if self.attention_head in ['1', 1]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head1.png")
            if self.attention_head in ['2', 2]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head2.png")
            if self.attention_head in ['3', 3]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head3.png")
            if self.attention_head in ['4', 4]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head4.png")
            if self.attention_head in ['5', 5]:
                image_name = os.path.join(self.data_folder, img_folder, img_name, "attn-head5.png")
        
        elif self.task in ['CM', 'Calc', 'Mass', 'CMMD_Malignant', 'CM_Malignant']:
            img_filename = self.mammography.iloc[idx, img_column]
            img_folder = img_filename.split("/")[-1][:-4]
            if self.rage:
                self.attention_head=random.randint(0,5)
            if self.attention_head in ['0', 0]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head0.png")
            if self.attention_head in ['1', 1]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head1.png")
            if self.attention_head in ['2', 2]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head2.png")
            if self.attention_head in ['3', 3]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head3.png")
            if self.attention_head in ['4', 4]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head4.png")
            if self.attention_head in ['5', 5]:
                image_name = os.path.join(self.data_folder, img_folder,  "attn-head5.png")

        

        image = cv2.imread(image_name)
        image = self.convert_to_mask(image, threshold = 45)
        image = cv2.resize(image, (448,448))
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.task in ['BreastDensity', 'MassRest', 'MassNormal']:
            image = self.image_load(idx, 2, 0) 
            mask = self.mask_load(idx, 2, 0)

        elif self.task in ['CM']:
            image = self.image_load(idx, 15, 18) 
            mask = self.mask_load(idx, 15, 18)

        elif self.task in ['CM_Malignant']:
            image = self.image_load(idx, 12, 18) 
            mask = self.mask_load(idx, 12, 18)

        elif self.task in ['Calc','Mass']:
            image = self.image_load(idx, 13, 0) 
            mask = self.mask_load(idx, 13, 0)

        elif self.task in ['CMMD_Malignant']:
            image = self.image_load(idx, 9, 0) 
            mask = self.mask_load(idx, 9, 0)

        if (torch.rand(1) < self.probability) and self.split == 'train' and self.task in ['CM', 'BreastDensity', 'MassRest', 'MassNormal']:
            mask = cv2.bitwise_not(mask)
            image = cv2.bitwise_and(image, image, mask=mask)
        elif (torch.rand(1) < self.probability) and self.split == 'train' and self.task in ['Calc', 'Mass', 'CMMD_Malignant', 'CM_Malignant']:
            #mask = cv2.bitwise_not(mask)
            image = cv2.bitwise_and(image, image, mask=mask)
        image = torch.from_numpy(image)  
        image = image.permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)

        label = self.class_name_to_labels(idx)
        labels = torch.from_numpy(np.array(label))
        

        return image, labels



class MammographyDatasetViz(Dataset):

    def __init__(self, split_file, data_folder, task, transform=None):

        self.mammography = pd.read_csv(split_file) 
        self.data_folder = data_folder
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self, idx):
        if self.task == 'BreastDensity':
            class_name = self.mammography.iloc[idx, 8]
            if class_name in ['DENSITY A']:
                labels = 0.0
            elif class_name in ['DENSITY B']:
                labels = 1.0
            elif class_name in ['DENSITY C']:
                labels = 2.0
            elif class_name in ['DENSITY D']:
                labels = 3.0
            return labels

        elif self.task == 'BIRADS':
            class_name = self.mammography.iloc[idx, 7]
            if class_name in ['BI-RADS 1']:
                labels = 0.0
            elif class_name in ['BI-RADS 2']:
                labels = 1.0
            elif class_name in ['BI-RADS 3']:
                labels = 2.0
            elif class_name in ['BI-RADS 4']:
                labels = 3.0
            elif class_name in ['BI-RADS 5']:
                labels = 4.0
            return labels

    def image_load(self, idx, img_column, folder_column):
        img_name = self.mammography.iloc[idx, img_column]
        study_id = self.mammography.iloc[idx, folder_column]
        image_name = os.path.join(self.data_folder, study_id, img_name)
        image_name = image_name + '.png'
        image = cv2.imread(image_name)
        #image = cv2.imresize(image, (512,512))
        image = torch.from_numpy(image)  
        #image = image.transpose(2, 0 , 1)
        image = image.permute(2, 0, 1)
        return image, study_id, img_name

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, study_id, img_name = self.image_load(idx, 2, 0) # 0 image folder, 1, 2nd column:image name
        #plt.imshow(image[0])
        #plt.savefig('image.png')
        if self.transform:
            image = self.transform(image)
        #plt.imshow(image[0][0])
        #plt.savefig('transformimage.png')

        label = self.class_name_to_labels(idx)
        labels = torch.from_numpy(np.array(label))
        

        return image, labels, study_id, img_name



'''
Example usage 

train_file = 'train.csv'
valid_file = 'validation.csv'
test_file = 'test.csv'

data_folder = '/cluster/eq27ifuw/dataset_png'

task = 'BreastDensity' ## Alternatively 'BIRADS'

train_transforms = torch.transforms(........)
test_transforms = torch.transforms(........)


train_dataset = MammographyDataset(split_file = train_file, data_folder = data_folder, task = task, transform=train_transforms)
valid_dataset = MammographyDataset(split_file = valid_file, data_folder = data_folder, task = task, transform=test_transforms)
test_dataset = MammographyDataset(split_file = test_file, data_folder = data_folder, task = task, transform=test_transforms)

'''