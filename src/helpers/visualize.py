import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os



def plot_groundtruth_prediction(city_title:str, groundtruth_path:str, prediction_path:str, save_path:str):

    groundtruth = np.load(groundtruth_path)
    prediction = np.load(prediction_path)

    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(city_title, fontsize=16)

    ax[0].set_title("Labels")
    ax[0].imshow(groundtruth)
    ax[0].set_axis_off()
    ax[1].set_title("Prediction")
    ax[1].imshow(prediction)
    ax[1].set_axis_off()

    f.tight_layout()

    plt.savefig(save_path + "/groundtruth_pred_" + city_title + ".png")

    return f



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
        plt.savefig(os.path.join(save_path,"ConfusionMatrix.png"))
