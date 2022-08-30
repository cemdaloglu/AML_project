import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def test_model(test_loader, criterion):
    best_path = 'best_unet.pth'
    model = torch.load(best_path)

    # evaluate on test set
    model = model.eval()

    test_loss_arr = []
    with torch.no_grad():
        epoch_test_loss = 0
        acc = 0
        pre = 0
        recall = 0
        f1 = 0
        conf_matrix = np.zeros((5, 5))
        #   iterate over test batches
        for loaded_data in test_loader:
            input_data, label = loaded_data
            input_data = input_data.to('cuda', dtype=torch.float32)
            label = label.to('cuda', dtype=torch.long)

            test_output = model(input_data.float())
            test_loss = criterion(test_output, label)

            epoch_test_loss += test_loss / len(test_loader)
            test_output = (test_output.argmax(dim=1)).long()
            label = np.array(label.cpu())
            test_output = np.array(test_output.cpu())
            #   get confusion matrix
            if (confusion_matrix(label, test_output).shape == (5, 5)):
                conf_matrix += confusion_matrix(label, test_output) / len(test_loader)
            #         conf_matrix.append(confusion_matrix(label, test_output))
            #   calculate accuracy
            acc += accuracy_score(label, test_output) / len(test_loader)
            #   calculate precision
            pre += precision_score(label, test_output, average='macro', zero_division=1) / len(test_loader)
            #   calculate recall
            recall += recall_score(label, test_output, average='macro', zero_division=1) / len(test_loader)
            #   calculate F1 score
            f1 += f1_score(label, test_output, average='macro') / len(test_loader)

    test_loss_arr.append(epoch_test_loss)
    test_loss_arr = np.array(test_loss_arr, dtype='float')
    losses = np.mean(test_loss_arr)

    # print metrics
    print("Mean Loss:", losses, "\nMean Acc:", acc, "\nMean Macro Precision:", pre, "\nMean Macro Recall:", recall,
          "\nMean Macro F1 Score:", f1)

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))

    fig.tight_layout()
    plt.show()