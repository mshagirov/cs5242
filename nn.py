# useful functions for model training and saving, etc.

__all__ = ['train_model']

import time
import copy
import torch

def train_model(model,
                loss_func,
                optimizer,
                data_loaders,
                device=torch.device('cpu'),
                num_epochs=5, scheduler=None):
    '''
    Multiclass classifier (single label) trainer.
    
    Arg-s:
    - model : model to be trained (an already initialized torch.nn.Module object)
    - loss_func: loss function (to be used as a criterion)
    - optimizer: optimizer (e.g. torch.optim.SGD)
    - data_loaders: data loaders (dict of torch.utils.data.DataLoader objects) with
                    keys 'train' and 'val', for training and validation data loaders respectively.
    - device: torch.device, either CUDA or CPU {default: torch.device('cpu')}.
    - num_epochs: total number of epochs to train.
    - scheduler: learning rate scheduler (from torch.optim.lr_scheduler)
    
    '''
    # model states/modes
    model_states = ['train', 'val']
    
    time_start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}'+'-'*10)
        
        # set model state depending on training/eval stage
        for state in model_states:
            if state == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[state]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(state == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler!=None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - time_start
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
