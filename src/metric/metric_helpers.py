import pandas as pd 
import csv

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def create_metric_file(path):
    metrics = pd.DataFrame(columns=["Epoch","Phase","Loss", "Accuracy", "F1Score", "Precision", "Recall"])
    metrics.to_csv(path, index=False)


def save_metrics(checkpoint_path_metrics, epoch, phase, loss, accuracy, f1Score, precision, recall):
    all_metrics = pd.read_csv(checkpoint_path_metrics + "metrics.csv")
    epoch_metrics = {'Epoch': epoch,
                     'Phase': phase,
                     'Loss': loss.cpu().numpy(),
                     'Accuracy': accuracy.cpu().numpy(),
                     'F1Score': f1Score.cpu().numpy(),
                     'Precision': precision.cpu().numpy(),
                     'Recall': recall.cpu().numpy()
                     }
    all_metrics = all_metrics.append(epoch_metrics, ignore_index=True)
    all_metrics.to_csv(checkpoint_path_metrics + "metrics.csv", index=False)
