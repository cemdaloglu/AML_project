""" Module for model training """

import time
import torch
from tqdm import tqdm
from ..metric.loss import calc_loss, init_loss
from ..metric.metric_helpers import save_metrics
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection

from src.helpers.visualize import plot_training


def train_model(model, dataloaders, use_cuda, optimizer, num_epochs, checkpoint_path_model, loss_criterion: str,
                trained_epochs: int = 0, freeze_epochs: int = 0, tb_writer = None, checkpoint_path_metrics = None):
    best_loss = 1e10
    total_acc = {key: [] for key in ['train', 'val']}
    total_loss = {key: [] for key in ['train', 'val']}
    loss_fn = init_loss(loss_criterion, use_cuda)
    since = time.time()

    metrics = MetricCollection([
        Accuracy(ignore_index=0, mdmc_average="global"),
        F1Score(ignore_index=0, mdmc_average="global"),
        Precision(ignore_index=0, mdmc_average="global"),
        Recall(ignore_index=0, mdmc_average="global")
    ])

    train_metrics = metrics.clone(prefix="train")
    val_metrics = metrics.clone(prefix="val")

    # iterate over all epochs
    for epoch in range(trained_epochs, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Unfreeze the frozen layers, if the chosen number of epochs has been reached
        if (
            freeze_epochs == epoch and
            hasattr(model, "frozen") and
            hasattr(model, "unfreeze_pretrained_params") and
            model.frozen == True
        ):
            model.unfreeze_pretrained_params()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0

            for dic in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs, labels = dic['image'], dic['mask']

                if use_cuda:
                    inputs = inputs.to('cuda', dtype=torch.float)  # [batch_size, in_channels, H, W]
                    labels = labels.to('cuda', dtype=torch.long)

                optimizer.zero_grad()  # zero the parameter gradients

                # forward pass: compute prediction and the loss btw prediction and true label
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # output is probability [batch size, n_classes, H, W], target is class [batch size, H, W]
                    loss = calc_loss(labels, outputs, loss_fn)

                    # backward + optimize only if in training phase (no need for torch.no_grad in this training pass)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss * outputs.size(0)
                preds_cpu = outputs.argmax(dim=1).cpu()
                labels_cpu = labels.cpu()
                if phase == "train":
                    train_metrics.update(preds_cpu, labels_cpu)
                elif phase == "val":
                    val_metrics.update(preds_cpu, labels_cpu)

            if phase == "train":
                computed_metrics = train_metrics.compute()
                train_metrics.reset()
            elif phase == "val":
                computed_metrics = val_metrics.compute()
                val_metrics.reset()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            computed_metrics[f"{phase}Loss"] = epoch_loss

            epoch_summary = f'Epoch {phase} : {epoch+1}'
            for k, v in computed_metrics.items():
                epoch_summary = f"{epoch_summary}\n\t{k} : {v.item():.6f}"

            print(epoch_summary)

            total_acc[phase].append(computed_metrics[f"{phase}Accuracy"].item())
            total_loss[phase].append(computed_metrics[f"{phase}Loss"].item())

        
            # Display metrics in Tensorboard
            if tb_writer is not None:
                for item in ["Loss", "Accuracy", "F1Score", "Precision", "Recall"]:
                    tb_writer.add_scalar(f"{item}/{phase}", computed_metrics[f"{phase}{item}"], epoch)

            if checkpoint_path_metrics is not None:
                metrics_list = []
                for item in ["Loss", "Accuracy", "F1Score", "Precision", "Recall"]:
                    metr = computed_metrics[f"{phase}{item}"]
                    metrics_list.append(metr)

                save_metrics(checkpoint_path_metrics, epoch, phase, *metrics_list)

            # save the model weights in validation phase 
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

        # Display total time
        time_elapsed = time.time() - since
        print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    plot_training(total_loss, total_acc)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))

    return model
