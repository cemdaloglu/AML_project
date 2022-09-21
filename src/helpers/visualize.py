import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os



def plot_groundtruth_prediction(city_title:str, groundtruth_path:str, prediction_path:str, save_path:str):

    groundtruth = np.load(groundtruth_path)
    prediction = np.load(prediction_path)

    f, ax = plt.subplots(1, 2, figsize=(20, 12))
    f.suptitle(city_title, fontsize=16)

    ax[0].set_title("Labels")
    ax[0].imshow(groundtruth)
    ax[0].set_axis_off()
    ax[1].set_title("Prediction")
    ax[1].imshow(prediction)
    ax[1].set_axis_off()

    f.tight_layout()

    plt.savefig(save_path + "/groundtruth_pred_" + city_title + ".png", bbox_inches=None)

    return f



def plot_groundtruth_bestpred_differences(city_title:str, best_model_name:str, model_name_list:list, img_groundtruth_pred_path:str):
    '''
    @param city_title: Heidelberg or Frankfurt 
    @param best_model_name: which model performed best to show the prediction
    @param img_groundtruth_pred_path: path to where image, groundtruth and all models with their predictions lie. 

    Output: 
        saves a plot with 6 subplots: groundtruth mask, best prediction mask, U-Net difference, 
                                    VGG16 difference, VGG pretrained difference and VGG index difference" 


    '''
    if city_title.casefold() == "Heidelberg".casefold(): 
        city_ind = 1
        city_name = "Heidelberg"
    elif (city_title.casefold() == "Frankfurt".casefold()) or (city_title.casefold() == "frankfurt am main".casefold()): 
        city_ind = 0
        city_name = "Frankfurt"
    
    groundtruth = np.load(os.path.join(img_groundtruth_pred_path, 'groundtruth_'+ str(city_ind)+".npy"))

    f, ax = plt.subplots(2, 3, figsize=(20, 15))
    #f.suptitle(city_title, fontsize=20)
    name_list = ["U-Net", "VGG16", "VGG16 pretrained", "VGG16 index"]

    best_prediction = np.load(os.path.join(img_groundtruth_pred_path, best_model_name, "pred_restored_"+str(city_ind)+".npy"))

    ax[0,0].set_title("Groundtruth", fontsize=30)
    ax[0,0].imshow(groundtruth)
    ax[0,0].set_axis_off()
    ax[0,1].set_title("Best Prediction", fontsize=30)
    ax[0,1].imshow(best_prediction)
    ax[0,1].set_axis_off()

    colors = ['black','white']
    diff = np.load(os.path.join(img_groundtruth_pred_path, model_name_list[0], "difference_"+str(city_ind)+".npy" ))
    ax[0,2].set_title(name_list[0], fontsize=30)
    ax[0,2].imshow(diff, cmap=matplotlib.colors.ListedColormap(colors))
    ax[0,2].set_axis_off()

    for (ind, model) in zip(range(1,len(model_name_list)), model_name_list[1:]):
        diff = np.load(os.path.join(img_groundtruth_pred_path, model, "difference_"+str(city_ind)+".npy" ))
        ax[1,ind-1].set_title(name_list[ind], fontsize=30)
        ax[1,ind-1].imshow(diff, cmap=matplotlib.colors.ListedColormap(colors))
        ax[1,ind-1].set_axis_off()

    f.tight_layout()

    print("saving to: ", img_groundtruth_pred_path + "/groundtruth_bestpred_diff_" + city_name + ".png")
    plt.savefig(img_groundtruth_pred_path + "/groundtruth_bestpred_diff_" + city_name + ".png", bbox_inches=None)

    return f



def plot_best_worst_segmentations(img_groundtruth_pred_path:str, best_model_name:str):
    '''
    @param best_model_name: which model performed best to show the prediction
    @param img_groundtruth_pred_path: path to where image, groundtruth and all models with their predictions lie. 

    Output: 
        saves a plot with 10 subplots: The best 5 segmentations and the worst 5 segmentations.

    '''
    best_worst_img_path = os.path.join(img_groundtruth_pred_path,best_model_name, "best_worst_imges")
    print(best_worst_img_path)
    n_plots = len(os.listdir(best_worst_img_path))
    
    #f, ax = plt.subplots(2, n_plots, figsize=(20, 15))

    fig = plt.figure(constrained_layout=True)
    # fig.suptitle('Figure title')            # set global suptitle if desired

    (best_plt, worst_plt) = fig.subfigures(2, 1) # create 2x1 subfigures
    ax1 = best_plt.subplots(1, n_plots)        
    ax2 = worst_plt.subplots(1, n_plots)       

    best_plt.suptitle('Best segmentations')               # set suptitle for subfig1
    worst_plt.suptitle('Worst segmentations')               # set suptitle for subfig2
    
    for ind in range(n_plots):
        best_pred = np.load(os.path.join(best_worst_img_path, "best_pred_" + str(ind) +".npy"))
        worst_pred = np.load(os.path.join(best_worst_img_path, "worst_pred_" + str(ind) +".npy"))
    #    
        #ax1[0,ind].set_title(name_list[0], fontsize=30)
        ax1[0,ind].imshow(best_pred)
        ax1[0,ind].set_axis_off()
        #ax2[1,ind].set_title(name_list[0], fontsize=30)
        ax2[1,ind].imshow(worst_pred)
        ax2[1,ind].set_axis_off()
        

    

    #for ind in range(n_plots):
    #    best_pred = np.load(os.path.join(best_worst_img_path, "best_pred_" + str(ind) +".npy"))
    #    worst_pred = np.load(os.path.join(best_worst_img_path, "worst_pred_" + str(ind) +".npy"))
    #    
    #    np.load(os.path.join(img_groundtruth_pred_path, model_name_list[0], "difference_"+str(city_ind)+".npy" ))
    #    ax[0,ind].set_title(name_list[0], fontsize=30)
    #    ax[0,ind].imshow(best_pred, cmap=matplotlib.colors.ListedColormap(colors))
    #    ax[0,ind].set_axis_off()
    #    ax[1,ind].set_title(name_list[0], fontsize=30)
    #    ax[1,ind].imshow(worst_pred, cmap=matplotlib.colors.ListedColormap(colors))
    #    ax[1,ind].set_axis_off()


    fig.tight_layout()

    print("saving to: ", best_worst_img_path + "/best_worst_img.png")
    plt.savefig(best_worst_img_path + "/best_worst_img.png", bbox_inches=None)

    return fig




def plot_image_groundtruth_prediction(image, groundtruth, prediction, loss = None):
    if torch.is_tensor(image):
        image = torch.permute(image[:,:,:3], (1, 2, 0)).numpy()
    if torch.is_tensor(groundtruth):
        groundtruth = groundtruth.numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.numpy()
    f, ax = plt.subplots(1, 3, figsize=(15, 7))
    ax[0].set_title("Image")
    ax[0].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_title("Labels")
    ax[1].imshow(groundtruth)
    ax[1].set_axis_off()
    ax[2].set_title("Prediction")
    ax[2].imshow(prediction)
    ax[2].set_axis_off()
    if loss is not None:
        f.suptitle(f"loss={loss:.2f}")
    f.tight_layout()
    return f


def plot_patches_with_masks(data, data_indices): 
    n_imgs = len(data_indices)

    f, ax = plt.subplots(n_imgs, 2, figsize = (8, 14))
    for i, ind in zip(range(n_imgs + 1), data_indices):
        sample = data[ind]
        
        plt.tight_layout()
        ax[i, 0].imshow(sample['image'][:,:,:3])
        ax[i, 1].imshow(sample['mask'])

        if i == n_imgs-1:
            plt.show()
            break


def plot_training(total_loss, total_acc):
    plt.plot(total_loss["train"], color='blue')
    plt.plot(total_loss["val"], color='orange')
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'valid_loss'])
    plt.show()

    plt.plot(total_acc["train"], color='blue')
    plt.plot(total_acc["val"], color='orange')
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'val_acc'])
    plt.show()


def plot_test(metrics, save_path = None):
    classes = ["unknown", "city", "agriculture", "natural", "wetlands", "water"]

    # print metrics
    global_summary = "Global:"
    for m in ["GlobalAccuracy", "GlobalF1Score", "GlobalPrecision", "GlobalRecall"]:
        global_summary = f"{global_summary}\n\t{m} : {metrics[m]:.6f}"

    per_class_summary = f"Per Class ({classes}):"
    for m in ["PerClassAccuracy", "PerClassF1Score", "PerClassPrecision", "PerClassRecall"]:
        per_class_summary = f"{per_class_summary}\n\t{m} : {metrics[m]}"

    print(global_summary)
    print(per_class_summary)

    # plot confusion matrix
    df_cm = pd.DataFrame(metrics["ConfusionMatrix"].numpy(), index=classes,
                         columns=classes)
    plt.figure(figsize=(12, 12))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt='.3g', annot_kws={'fontsize': 14})

    if save_path is not None:
        print("saving confusion matrix to: ", save_path)
        plt.savefig(os.path.join(save_path,"ConfusionMatrix.png"), bbox_inches='tight')
