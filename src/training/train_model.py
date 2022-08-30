""" Module for model training """

import time
import torch
from tqdm import tqdm
from ..metric.loss import calc_loss
import matplotlib.pyplot as plt


def train_model(model, dataloaders, use_cuda, optimizer, num_epochs, checkpoint_path_model, trained_epochs=0):
    best_loss = 1e10

    # iterate over all epochs
    for epoch in range(trained_epochs, num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_samples = 0

            for dic in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs, labels = dic['image'], dic['mask']

                if use_cuda:
                    inputs = inputs.to('cuda', dtype=torch.float)  # [batch_size, in_channels, H, W]
                    labels = labels.to('cuda', dtype=torch.float)

                optimizer.zero_grad()  # zero the parameter gradients

                epoch_loss = 0
                # forward pass: compute prediction and the loss btw prediction and true label
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # output is probability [batch size, n_classes, H, W], target is class [batch size, H, W]
                    # TODO: decide on loss!! (dummy function here)
                    loss = calc_loss(outputs, labels.long())

                    # backward + optimize only if in training phase (no need for torch.no_grad in this training pass)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss

                # statistics
                epoch_samples += inputs.size(0)

            epoch_loss = loss / epoch_samples
            print("epoch_loss = ", epoch_loss)

            # save the model weights in validation phase 
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))
    return model


# start training
# for each epoch calculate validation performance
# save best model according to validation performance
def train_model_better(model, optimizer, num_epochs, train_loader, val_loader, criterion):

    val_acc = []
    valid_loss = []
    train_acc = []
    train_loss = []
    val_acc_max = 0

    for epoch in range(num_epochs):
        model = model.train()
        epoch_accuracy = 0
        epoch_loss = 0
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to('cuda', dtype=torch.float32)
            labels = labels.to('cuda', dtype=torch.long)

            optimizer.zero_grad()

            inputs = inputs.permute(0, 3, 1, 2)  # maybe not needed, depends on data loader i guess
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = ((outputs.argmax(dim=1) == labels).float().mean())
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        train_acc.append(epoch_accuracy)
        train_loss.append(epoch_loss)
        #    Validation
        model = model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data in val_loader:
                data, label = data
                data = data.to('cuda', dtype=torch.float32)
                label = label.to('cuda', dtype=torch.long)

                data = data.permute(0, 3, 1, 2)  # maybe not needed, depends on data loader i guess
                val_output = model(data.float())
                val_loss = criterion(val_output, label)

                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)
            val_acc.append(epoch_val_accuracy)
            valid_loss.append(epoch_val_loss)
            if epoch_val_accuracy > val_acc_max:
                torch.save(model, 'best_unet.pth')
                val_acc_max = epoch_val_accuracy

    plt.plot(train_loss, color='green')
    plt.plot(valid_loss, color='red')
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'valid_loss'])
    plt.show()

    plt.plot(train_acc, color='green')
    plt.plot(val_acc, color='red')
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train_acc', 'val_acc'])
    plt.show()
