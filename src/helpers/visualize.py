import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


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
    total_loss_train = []
    for idx in range(len(total_loss['train'])):
        total_loss_train.append(total_loss['train'][idx].detach().cpu().numpy())

    total_loss_val = []
    for idx in range(len(total_loss['val'])):
        total_loss_val.append(total_loss['val'][idx].detach().cpu().numpy())

    total_acc_train = []
    for idx in range(len(total_acc['train'])):
        total_acc_train.append(total_acc['train'][idx].detach().cpu().numpy())

    total_acc_val = []
    for idx in range(len(total_acc['val'])):
        total_acc_val.append(total_acc['val'][idx].detach().cpu().numpy())

    plt.plot(total_loss_train, color='blue')
    plt.plot(total_loss_val, color='orange')
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'valid_loss'])
    plt.show()

    plt.plot(total_acc_train, color='blue')
    plt.plot(total_acc_val, color='orange')
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'val_acc'])
    plt.show()


def plot_test(test_losses, acc, pre, recall, f1, conf_matrix, num_correct, num_pixels):
    # print metrics
    print("Mean Loss:", np.mean(test_losses), "\nMean Acc:", acc, "\nMean Macro Precision:", pre,
          "\nMean Macro Recall:", recall,
          "\nMean Macro F1 Score:", f1)

    # plot confusion matrix
    df_cm = pd.DataFrame(conf_matrix, index=["unknown", "agriculture", "forest", "city", "wetlands", "water"],
                         columns=["unknown", "agriculture", "forest", "city", "wetlands", "water"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    test_accuracy = (num_correct / num_pixels * 100).cpu().numpy()
    print(f"Got {num_correct}/{num_pixels} with acc {test_accuracy}")