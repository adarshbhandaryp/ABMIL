import torch
import pandas as pd
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

def log_scalar_values(file_path : str, data_dict : dict):
    ## Create a CSV file with results
    try:
        check = pd.read_csv(file_path)
        df = pd.DataFrame(data_dict, index=[0])
        df.to_csv(file_path, mode='a', index=False, header=False)
    except:
        df = pd.DataFrame(data_dict, index=[0])
        df.to_csv(file_path, index=False, header = True)

def count_class_distribution(file, which_labels):
    ## Old Code. Check
    read_file = pd.read_excel(file)
    labels = read_file[which_labels].tolist()
    values, counts = np.unique(labels, return_counts=True)
    return counts[::-1]


def set_seed(seed):
    ## Random Seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def label_encoder(labels, class_type = 'subtype_class'):
    ## Keeping this for easier understanding of the labels. Not needed; Only when labels are strings
    
    if class_type in ['Calc_Mass_Malignant_MultiLabel']:
        if labels == 'calcification_Benign':
            label = 0
        elif labels == 'calcification_Malignant':
            label = 1    
        elif labels == 'mass_Benign' :
            label = 2
        elif labels == 'mass_Malignant':
            label = 3
        elif labels == 'both_Benign':
            label = 4
        elif labels == 'both_Malignant':
            label = 5

    elif class_type in ['CMMD_Malignant']:
        if labels == 'Benign':
            label = 0
        elif labels == 'Malignant':
            label = 1    

    elif class_type in ['2i', '3i', 'Skinfolds', 'Skinfold_MultiLabel', 'Skinfold_Defect_MultiLabel']:
        if labels == 0 or labels == '0':
            label = 0
        elif labels == 1 or labels == '1':
            label = 1

    elif class_type in ['No Defects']:
        if labels == 0 or labels == '0':
            label = 1
        elif labels == 1 or labels == '1':
            label = 0

    elif class_type in ['BreastDensity']:
        if labels in ['DENSITY A']:
            label = 0
        elif labels in ['DENSITY B']:
            label = 1
        elif labels in ['DENSITY C']:
            label = 2
        elif labels in ['DENSITY D']:
            label = 3

    elif class_type in ['CM', 'Calc', 'Mass', 'CM_Malignant']:
        if labels in ['Normal', 'BENIGN_WITHOUT_CALLBACK']:
            label = 0
        elif labels in ['Benign', 'BENIGN']:
            label = 1
        elif labels in ['Malignant', 'MALIGNANT']:
            label = 2

    elif class_type in ['MassRest']:
        if 'Mass' in labels:
            label = 0
        else:
            label = 1

    elif class_type in ['MassNormal']:
        if 'Mass' in labels:
            label = 0
        elif 'No Finding' in labels:
            label = 1
        else:
            label = 2

    else:
        raise ValueError("Problem in label_encoder in util functions. Check the method")
    return label
    
def compute_sample_weights(csv_file:str, class_type:str, view:str):
    ## Compute sample weights and weights for Weighted Random Sampler and Class-Weighted Cross Entropy
    read_csv = pd.read_csv(csv_file)
    if view in ['CC','MLO']:
        read_csv = read_csv[read_csv['View Position'] == view]
    if class_type in ['Calc_Mass_Malignant_MultiLabel']:
        abnormality = read_csv['abnormality']
        classification = read_csv['classification']
        labels = abnormality + '_' + classification
    elif class_type in ['CMMD_Malignant']:
        labels = read_csv['classification']
    elif class_type in ['MassRest', 'MassNormal']:
        labels = read_csv['finding_categories']
    elif class_type in ['2i']:
        labels = read_csv['2i']
    elif class_type in ['3i']:
        labels = read_csv['3i']
    elif class_type in ['Skinfolds']:
        labels = read_csv['Skinfolds']
    elif class_type in ['No Defects']:
        labels = read_csv['no_defect']
    elif class_type in ['Skinfold_MultiLabel', 'Skinfold_Defect_MultiLabel']:
        labels = read_csv['Skinfolds'] ## No better idea currently : TODO
    elif class_type in ['BreastDensity']:
        labels = read_csv['breast_density']


    elif class_type in ['CM', 'CM_Malignant']:
        labels = read_csv['Pathology Classification/ Follow up']

    elif class_type in ['Calc', 'Mass']:
        labels = read_csv['pathology']

    elif class_type in ['MRI', 'MRI_MIL']:
        read_csv = read_csv[read_csv['Split_patient'] == "train"].reset_index(drop=True)
        labels = read_csv['Lesion'].astype(str) + '_' + read_csv['Institution'].astype(str)

    targets = labels.tolist()
    if class_type not in ['MRI', 'MRI_MIL']:
        tar = [label_encoder(labels, class_type = class_type) for labels in targets]
        target = np.array(tar)
        class_sample_count = np.unique(target, return_counts=True)[1]
        print('Class Sample Count : ', class_sample_count)
        class_weights=compute_class_weight(class_weight ='balanced', classes = np.unique(tar),y = tar)
        class_weights_torch=torch.tensor(class_weights,dtype=torch.float)
        print('Class Weights: ', class_weights)
        samples_weight = class_weights[target]
        return class_weights_torch, samples_weight

    else: 
        le = LabelEncoder()
        targets = le.fit_transform(labels)

        # 4) Compute how many samples per class
        class_counts = np.bincount(targets)
        print("Class sample counts:", class_counts)

        # 5) Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(targets),
            y=targets
        )
        print("Class weights:", class_weights)

        # 6) Torch tensor of per-class weights
        class_weights_torch = torch.tensor(class_weights, dtype=torch.float)

        # 7) Per-sample weights array
        samples_weight = class_weights[targets]

        return class_weights_torch, samples_weight



def labels_for_classification(labels):
    if labels[0].detach().cpu().numpy() == 0:
        title = 'HRS Positive'
    elif labels[0].detach().cpu().numpy() == 1:
        title = 'HRS Negative'
    return title

def labels_for_classification2(labels):
    if labels[0].detach().cpu().numpy() == 0:
        title = 'Follow Up'
    elif labels[0].detach().cpu().numpy() == 1:
        title = 'No Follow Up'
    return title
    