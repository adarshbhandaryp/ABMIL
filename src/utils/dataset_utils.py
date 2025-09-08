
def extract_target_names(method):

    if method in ['Calc_Mass_Malignant_MultiLabel']:
        target_names = ['Calcification', 'Mass', 'Malignant']

    elif method in ['CMMD_Malignant']:
        target_names = ['Benign', 'Malignant']

    elif method in ['MassRest']:
        target_names = ['Mass', 'No Mass'] 
    
    elif method in ['MassNormal']:
        target_names = ['Mass', 'No Finding', 'Others'] 

    elif method in ['No Defects']:
        target_names = ['No Defects', 'Defects']   

    elif method in ['Skinfolds', '2i', '3i']:
        target_names = ['No Skinfolds', 'Skinfolds']  

    elif method in ['Skinfold_MultiLabel']:
        target_names = ['Skinfolds', 'Defects']  

    elif method in ['Skinfold_Defect_MultiLabel']:
        target_names = ['2i Skinfolds', '3i Skinfolds', 'Defects'] 

    elif method in ['BreastDensity']:
        target_names = ['Density A', 'Density B', 'Density C', 'Density D'] 

    elif method in ['CM', 'CM_Malignant']:
        target_names = ['Normal', 'Benign', 'Malignant'] 

    elif method in ['CM_Malignant_Binary']:
        target_names = ['Non-Malignant', 'Malignant'] 

    elif method in ['Calc']:
        target_names = ['Normal', 'Benign', 'Malignant'] 

    elif method in ['Mass']:
        target_names = ['Normal', 'Benign', 'Malignant'] 

    elif method in ['MRI', 'MRI_MIL']:
        target_names = ['Normal', 'Benign', 'Malignant'] 
    
    else:
        raise ValueError("Method not correctly specified")
    return target_names