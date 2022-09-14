""" Module for model evaluation """

import torch
from tqdm import tqdm
from ..metric.loss import calc_loss, init_loss
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection, ConfusionMatrix
# helper function to get the root path
from pathlib import Path
from ..helpers.visualize import plot_test
import seaborn as sns


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent


def test(model, test_loader, use_cuda: bool, loss_criterion: str, n_classes: int):
    """
    Compute test metrics on test data set 

    @param model: -- the neural network
    @param: use_cuda: -- true if GPU should be used
    @param: loss_fun: -- the used loss function from calc_loss
    @param: test_loader: -- test data dataloader
    @param: test_batch_size: -- used batch size
    """

    running_loss = 0

    model.eval()
    loss_fn = init_loss(loss_criterion, use_cuda)

    metrics = MetricCollection({
        "GlobalAccuracy": Accuracy(ignore_index=0, mdmc_average="global"),
        "PerClassAccuracy": Accuracy(ignore_index=0, mdmc_average="global",
                                     average="none", num_classes=n_classes),
        "GlobalF1Score": F1Score(ignore_index=0, mdmc_average="global"),
        "PerClassF1Score": F1Score(ignore_index=0, mdmc_average="global",
                                   average="none", num_classes=n_classes),
        "GlobalPrecision": Precision(ignore_index=0, mdmc_average="global"),
        "PerClassPrecision": Precision(ignore_index=0, mdmc_average="global",
                                       average="none", num_classes=n_classes),
        "GlobalRecall": Recall(ignore_index=0, mdmc_average="global"),
        "PerClassRecall": Recall(ignore_index=0, mdmc_average="global",
                                 average="none", num_classes=n_classes),
        "ConfusionMatrix": ConfusionMatrix(n_classes, normalize="true")
    })

    with torch.no_grad():
        for dic in tqdm(test_loader, total=len(test_loader)):
            inputs, labels = dic['image'], dic['mask']

            if use_cuda:
                inputs = inputs.to('cuda', dtype=torch.float)  # [batch_size, in_channels, H, W]
                labels = labels.to('cuda', dtype=torch.long)

            outputs = model(inputs)

            loss = calc_loss(labels, outputs, loss_fn)

            # statistics
            preds_cpu = outputs.argmax(dim=1).cpu()
            labels_cpu = labels.cpu()
            running_loss += loss * outputs.size(0)
            metrics.update(preds_cpu, labels_cpu)

            # TODO: Store predicted mask for visualization?
            ...

        computed_metrics = metrics.compute()

        test_loss = running_loss / len(test_loader.dataset)
        computed_metrics[f"Loss"] = test_loss

    # plot_test(computed_metrics)
    swarm_plot = sns.swarmplot(computed_metrics)
    fig = swarm_plot.get_figure()
    fig.savefig("computed_metrics.png")