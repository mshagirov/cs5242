# useful functions for model training and saving, etc.

__all__ = ['train_model']

import time
import copy
import torch

def train_model(model,
                optimizer,
                data_loaders,
                num_epochs = 5,
                loss_func = torch.nn.CrossEntropyLoss(),
                device = torch.device('cpu'),
                scheduler = None,
                return_best = False):
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
    - return_best: return the best model in the end of the training instead of the latest model
                  (performance is measured on the validation set) {default: False}. Best validation set
                  accuracy is always measured and tracked regardless of this argument.
    - returns: model, and dict of losses and accuracies ("curve_data").
    '''
    # model states/modes
    model_states = ['train', 'val']
    
    curve_data = {'trainLosses':[],
                 'trainAccs':[],
                 'valLosses':[],
                 'valAccs':[],
                 'total_epochs':num_epochs}
    
    time_start = time.time()
    if return_best:
        best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1} ---', end=' ')
        
        # set model state depending on training/eval stage
        for state in model_states:
            if state == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            for samples in data_loaders[state]:
                # input HxW depend on transform function(s), 3 Channels
                inputs = samples['image'].to(device)
                # labels \in [0, 1, 2]
                labels = samples['label'].to(device)          
                
                # set grad accumulator to zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    # grad tracking is disabled in "eval" mode
                    outputs = model(inputs) # output:(batch, #classes)
                    _, preds = torch.max(outputs, 1) # labels:(batch,)
                    loss = loss_func(outputs, labels) #<-torch.nn.CrossEntropyLoss
                    
                    if state == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) # weighted loss
                running_corrects += torch.sum(preds == labels.detach() )
            
                # apply LR schedule
                if state == 'train' and scheduler!=None:
                    scheduler.step()

            epoch_loss = running_loss / len(data_loaders[state].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[state].dataset)
            
            curve_data[f'{state}Losses'].append(epoch_loss)
            curve_data[f'{state}Accs'].append(epoch_acc)
            
            print(f'{state} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}',end=' || ')
            
            # deep copy the model
            if state == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if return_best:
                    # keep best weights to return
                    best_model_wts = copy.deepcopy(model.state_dict())
        print(f'{time.time() - time_start:.0f}s')
    time_elapsed = time.time() - time_start
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} (return best:{return_best})')
    
    if return_best:
        # load best model weights
        model.load_state_dict(best_model_wts)
    
    return model, curve_data
