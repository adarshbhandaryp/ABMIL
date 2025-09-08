import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from src.utils.utils_functions import labels_for_classification, labels_for_classification2
from skimage import exposure

def display_class_report(classification_report, folder_output, use_tta, label ):
    sns.heatmap(pd.DataFrame(classification_report).iloc[:-1, :].T, annot=True)
    if use_tta:
        if label == 'classification':
            filename = folder_output + label + '_class_report_tta.png'
        else:
            filename = folder_output + '_class_report_tta.png'
    else:
        if label == 'classification':
            filename = folder_output + label + '_class_report.png'
        else:
            filename = folder_output + 'class_report.png'
    plt.savefig(filename)
    #plt.close()


def display_confusion_matrix(true, pred, display_labels, folder_output, use_tta, label, normalize=True):
    if normalize:
        cm = confusion_matrix(true, pred, normalize='true')
        if label == 'classification':
            filename = folder_output + str(use_tta) + label + '_normalized_confusion_matrix.png'
        else:
            filename = folder_output + str(use_tta) + '_normalized_confusion_matrix.png'
    else:
        cm = confusion_matrix(true, pred)
        if label == 'classification':
            filename = folder_output + str(use_tta) + label + '_confusion_matrix.png'
        else:
            filename = folder_output + str(use_tta) + '_confusion_matrix.png'
    plt.tight_layout()
    disp = ConfusionMatrixDisplay(cm)#, display_labels)
    disp.plot()
    
    plt.savefig(filename)
    plt.close('all')


def plot_roc_curve(filename, scores, labels):
    """plot_roc_curve plots (saves) roc curve for a binary classification scenario
    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
    """
    lw = 1

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(1, figsize=(10, 10))
    plt.plot(
        fpr,
        tpr,
        label="ROC curve (area = {0:0.2f})" "".format(roc_auc),
        color="green",
        linestyle="--",
        linewidth=2,
    )
    plt.rcParams.update({'font.size': 18})
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate",fontsize=18)
    #plt.title("Receiver Operating Characteristic")
    
    plt.tight_layout()
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_roc.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close('all')










def plot_roc_curve_multiclass(filename, scores, labels, classes):
    """plot_roc_curve_multiclass plots (saves) roc curve for a multiclass classification scenario
    Code from : https://github.com/icrto/xML/blob/master/PyTorch/utils.py
    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
        classes {list} -- list of class names
    """

    line_width = 3
    nr_classes = len(classes)
    labels = label_binarize(labels, classes=list(range(nr_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nr_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nr_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    colors = cycle(
        [
            "coral",
            "mediumorchid",
            "aqua",
            "darkolivegreen",
            "cornflowerblue",
            "gold",
            "pink",
            "chocolate",
            "brown",
            "darkslategrey",
            "tab:cyan",
            "slateblue",
            "yellow",
            "palegreen",
            "tan",
            "silver",
        ]
    )
    for i, color in zip(range(nr_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=line_width,
            label="class {0} (AUC = {1:0.4f})" "".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    lgd = plt.legend(loc="best")
    plt.savefig(
        filename + "_roc_all.png",  bbox_inches="tight"
    )
    plt.close()

    plt.figure(2, figsize=(10, 10))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average (AUC = {0:0.4f})" "".format(roc_auc["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average (AUC = {0:0.4f})" "".format(roc_auc["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_roc.png", bbox_inches="tight")
    plt.close()
    return roc_auc["macro"], roc_auc["micro"]

def plot_precision_recall_curve_multiclass(filename, scores, labels, classes):
    """plot_precision_recall_curve_multiclass plots (saves) precision vs recall curve for a multiclass classification scenario
    Code from : https://github.com/icrto/xML/blob/master/PyTorch/utils.py
    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
        classes {list} -- list of class names
    """
    line_width = 3
    nr_classes = len(classes)
    labels = label_binarize(labels, classes=list(range(nr_classes)))

    precision = dict()
    recall = dict()
    auc_prec_recall = dict()
    average_precision = dict()
    for i in range(nr_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], scores[:, i])
        auc_prec_recall[i] = auc(recall[i], precision[i])
        average_precision[i] = average_precision_score(labels[:, i], scores[:, i])

    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        labels.ravel(), scores.ravel()
    )
    auc_prec_recall["micro"] = auc(recall["micro"], precision["micro"])
    average_precision["micro"] = average_precision_score(
        labels, scores, average="micro"
    )

    # Compute macro-average
    # First aggregate all recall
    all_recall = np.unique(np.concatenate([recall[i] for i in range(nr_classes)]))

    # Then interpolate all ROC curves at these points
    mean_precision = np.zeros_like(all_recall)
    for i in range(nr_classes):
        mean_precision += np.interp(all_recall, recall[i], precision[i])

    # Finally average it and compute AUC
    mean_precision /= nr_classes

    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    auc_prec_recall["macro"] = auc(recall["macro"], precision["macro"])
    average_precision["macro"] = average_precision_score(
        labels, scores, average="macro"
    )

    # Plot all ROC curves
    plt.figure(1, figsize=(10, 10))
    plt.plot(
        recall["micro"],
        precision["micro"],
        label="micro-average (AP = {0:0.4f}; AUC = {0:0.4f})"
              "".format(average_precision["micro"], auc_prec_recall["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        recall["macro"],
        precision["macro"],
        label="macro-average (AP = {0:0.4f}; AUC = {0:0.4f})"
              "".format(average_precision["macro"], auc_prec_recall["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    colors = cycle(
        [
            "coral",
            "mediumorchid",
            "aqua",
            "darkolivegreen",
            "cornflowerblue",
            "gold",
            "pink",
            "chocolate",
            "brown",
            "darkslategrey",
            "tab:cyan",
            "slateblue",
            "yellow",
            "palegreen",
            "tan",
            "silver",
        ]
    )
    for i, color in zip(range(nr_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            line_width=line_width,
            label="class {0} (AP = {1:0.4f}; AUC = {1:0.4f})"
                  "".format(classes[i], average_precision[i], auc_prec_recall[i]),
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    lgd = plt.legend(loc="best")
    plt.savefig(
        filename + "_prec_recall_all.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(2, figsize=(10, 10))
    plt.plot(
        recall["micro"],
        precision["micro"],
        label="micro-average (AP = {0:0.4f}; AUC = {0:0.4f})"
              "".format(average_precision["micro"], auc_prec_recall["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        recall["macro"],
        precision["macro"],
        label="macro-average (AP = {0:0.4f}; AUC = {0:0.4f})"
              "".format(average_precision["macro"], auc_prec_recall["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], "k--", line_width=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_prec_recall.png", bbox_inches="tight")
    plt.close()


def plot_grad_cam(images, grad_image, count, labels, predicted, save_out_folder):
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    ax1.imshow(images[0][0].cpu().numpy(), cmap='gray')
    plt.title('True Label: ' + labels_for_classification(labels))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    
    ax2.imshow(grad_image)
    plt.title('Predicted Label: ' + labels_for_classification(predicted))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_out_folder+ labels_for_classification(labels)+'_'+ str(count) + '.png', dpi=600, bbox_inches='tight')
    #plt.savefig(save_out_folder+ labels_for_classification(labels)+'_'+ str(count) + '.eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_grad_cam_epoch(images, grad_image, count, i, labels, predicted, save_out_folder):
    plt.figure(figsize=(10, 10))
    #plt.imshow(images[0][0].cpu().numpy(), cmap='gray')
    #plt.title('True Label: ' + labels_for_classification2(labels))
    plt.imshow(grad_image)
    plt.title('Epoch: ' + str(i) )
    #plt.title('Predicted Label: ' + labels_for_classification2(predicted))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.rcParams.update({'font.size': 26})
    plt.savefig(save_out_folder+ 'GradCam_' + str(count)+'_'+ str(i+1)+ '.png', dpi=100)
    plt.close('all') 
    
    
def save_input_array_grad_array(images, grad_image, filename, save_out_folder):
    images = images[0][0].cpu().numpy()
    image_filename = os.path.join(save_out_folder,filename+'_image.npy')
    mask_filename = os.path.join(save_out_folder, filename+'_mask.npy')
    np.save(image_filename, images, allow_pickle=True, fix_imports=True)
    np.save(mask_filename, grad_image, allow_pickle=True, fix_imports=True)
    return image_filename, mask_filename
    
    
def plot_grad_cam_histogram(images, grad_image, count, labels, predicted, save_out_folder):
    #fig = plt.figure(figsize=(10, 10))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    ax1.imshow(images[0][0].cpu().numpy(), cmap='gray')
    #plt.title('True Label: ' + labels_for_classification2(labels))
    
    
    
    
    ax3 = fig.add_subplot(1, 3, 2)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    img = images[0][0].cpu().numpy()
    img_eq = exposure.equalize_hist(img)
    ax3.imshow(img_eq, cmap='gray')
    #plt.title('True Label: ' + labels_for_classification2(labels))
    
    
    
    
    
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.imshow(grad_image)
    #plt.title('Predicted Label: ' + labels_for_classification2(predicted))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_out_folder+ labels_for_classification(labels)+'_'+labels_for_classification(predicted)+ str(count) + '.jpg', dpi=800, bbox_inches='tight')
    plt.savefig(save_out_folder+ labels_for_classification(labels)+'_'+labels_for_classification(predicted)+ str(count) + '.eps', dpi=800, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_grad_cam_single(images, grad_image, count, labels, predicted, save_out_folder):
    #fig = plt.figure(figsize=(10, 10))
    
    fig = plt.figure()

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.imshow(grad_image)
    #plt.imshow(images[0][0].cpu().numpy(), cmap='gray')
    #img = images[0][0].cpu().numpy()
    #img_eq = exposure.equalize_hist(img)
    #plt.imshow(img_eq, cmap='gray')
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.savefig(save_out_folder+ labels_for_classification2(labels)+'_'+labels_for_classification2(predicted)+ str(count) + '.jpg', dpi=300, bbox_inches='tight')
    #plt.savefig(save_out_folder+ labels_for_classification2(labels)+'_'+labels_for_classification2(predicted)+ str(count) + '.eps', dpi=300, bbox_inches='tight')
    plt.savefig(save_out_folder+ labels_for_classification(labels)+ str(count) + '.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(save_out_folder+ labels_for_classification(labels)+ str(count) + '.eps', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_roc_curve_multiclass1(filename, scores, labels, classes):
    """plot_roc_curve_multiclass plots (saves) roc curve for a multiclass classification scenario
    Code from : https://github.com/icrto/xML/blob/master/PyTorch/utils.py
    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
        classes {list} -- list of class names
    """

    line_width = 3
    nr_classes = len(classes)
    labels = label_binarize(labels, classes=list(range(nr_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nr_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nr_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # colors = cycle(
        # [
            # "coral",
            # "mediumorchid",
            # "aqua",
            # "darkolivegreen",
            # "cornflowerblue",
            # "gold",
            # "pink",
            # "chocolate",
            # "brown",
            # "darkslategrey",
            # "tab:cyan",
            # "slateblue",
            # "yellow",
            # "palegreen",
            # "tan",
            # "silver",
        # ]
    # )
    # # for i, color in zip(range(nr_classes), colors):
        # # plt.plot(
            # # fpr[i],
            # # tpr[i],
            # color=color,
            # lw=line_width,
            # label="class {0} (AUC = {1:0.4f})" "".format(classes[i], roc_auc[i]),
        # )

    # plt.plot([0, 1], [0, 1], "k--", lw=line_width)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # lgd = plt.legend(loc="best")
    # plt.savefig(
        # filename + "_roc_all.png",  bbox_inches="tight"
    # )
    # # plt.close()

    # plt.figure(2, figsize=(10, 10))
    # plt.plot(
        # fpr["micro"],
        # tpr["micro"],
        # label="micro-average (AUC = {0:0.4f})" "".format(roc_auc["micro"]),
        # color="green",
        # linestyle="--",
        # linewidth=2,
    # )

    # plt.plot(
        # fpr["macro"],
        # tpr["macro"],
        # label="macro-average (AUC = {0:0.4f})" "".format(roc_auc["macro"]),
        # color="red",
        # linestyle=":",
        # linewidth=2,
    # )

    #plt.plot([0, 1], [0, 1], "k--", lw=line_width)
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title("Receiver Operating Characteristic")
    #lgd = plt.legend(loc="best")
    ##plt.close()
    
    return fpr["macro"],tpr["macro"],roc_auc["macro"]