import pandas as pd 

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def create_metric_file(path):
    metrics = pd.DataFrame(columns=["Epoch","Phase","Loss", "Accuracy", "F1Score", "Precision", "Recall"])
    metrics.to_csv(path, index=False)
