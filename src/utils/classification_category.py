def classification_task(category, cv):
    if category == 'Five_Class':
        num_classes = 6
        which_labels = 'Labels'
        target_names = ['MASS_Malignant', 'MASS_Benign', 'MASS_BWC', 'CALC_Malignant', 'CALC_Benign', 'CALC_BWC']
        excel_file_train = 'D:\\SSL_Breast_Lesion\\Labels\\Train_CV2.csv'
        excel_file_validation = 'D:\\SSL_Breast_Lesion\\Labels\\Valid_CV2.csv'
        excel_file_test = 'D:\\SSL_Breast_Lesion\\Labels\\Test_Full.csv'

    return num_classes, target_names, excel_file_train, excel_file_validation, excel_file_test


def pretraining_task(category):
    if category == 'Five_Class':
        num_classes = 6
        which_labels = 'Labels'
        target_names = ['MASS_Malignant', 'MASS_Benign', 'MASS_BWC', 'CALC_Malignant', 'CALC_Benign', 'CALC_BWC']
        excel_file_train = 'D:\\SSL_Breast_Lesion\\Labels\\Train_Full.csv'
        excel_file_validation = 'na'
        excel_file_test = 'na'

    return num_classes, target_names, excel_file_train, excel_file_validation, excel_file_test
