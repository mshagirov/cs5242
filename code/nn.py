# useful functions for model training and saving, etc.

__all__ = ['train_model', 'predict_check', 'predict_test']

import time
import copy
import torch
from os import path
import glob
from skimage.io import imread

def train_model(model,
                optimizer,
                data_loaders,
                num_epochs = 5,
                loss_func = torch.nn.CrossEntropyLoss(),
                device = torch.device('cpu'),
                scheduler = None,
                return_best = False,
               classifier=None):
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
    training_model=model
    if classifier!=None:
        training_model=classifier # transfer learning
    
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
                training_model.train()  # Set model to training mode
            else:
                training_model.eval()   # Set model to evaluation mode
            
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


@torch.no_grad()
def predict_check(data_loaders, model,device=torch.device('cpu')):
    '''
    Run prediction on datasets using dataloader
    - data_loaders: data loaders (dict of torch.utils.data.DataLoader objects) with
                    keys 'train' and 'val', for training and validation data loaders respectively.
                    ! DISABLE SHUFFLING in both datasets in order to preserve order of IDs
    - model: model used for prediction
    - device: device, e.g. "torch.device('cuda')"
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    losses = {'train':0, 'val': 0}
    accuracies = {'train':0, 'val':0}
    pred_labels = {'train':[],'val':[]}
    
    model.eval()
    default_device = next(model.parameters()).device
    model.to(device)
    for loader_type in data_loaders:
        print(f'Loading: {loader_type}')
        for samples in data_loaders[loader_type]:
            inputs = samples['image'].to(device) # input images
            labels = samples['label'].to(device) # labels \in [0, 1, 2]       
            # predict
            outputs = model(inputs) # output:(batch, #classes)
            _, preds = torch.max(outputs, 1) # labels:(batch,)
            preds = preds.cpu()
            loss = loss_func(outputs, labels) #<-torch.nn.CrossEntropyLoss
            losses[loader_type] += loss.item() * inputs.size(0) # weighted loss
            accuracies[loader_type] += torch.sum(preds == labels.cpu() )
            pred_labels[loader_type].extend(preds.tolist())
            
        losses[loader_type] = losses[loader_type] / len(data_loaders[loader_type].dataset)
        accuracies[loader_type] = accuracies[loader_type] / len(data_loaders[loader_type].dataset)
    print('Losses:',losses)
    print('Accuracies:', accuracies)
    
    model.to(default_device)
    return losses, accuracies, pred_labels


@torch.no_grad()
def predict_test(root_path, model, transform, batch_size=4, device=torch.device('cpu')):
    '''Run prediction on test images.
    - root_path: path to the folder with test images
    - model: model used for prediction 
    - batch_size: batch size for processing
    - device: device, e.g. "torch.device('cuda')"
    '''
    model.eval()
    model.to(device)
    # list of test image files
    test_image_names= [path.split(imgname)[-1] for imgname in glob.glob(path.join(root_path,'*.png'))]
    # sort according to image ID number
    test_image_names.sort(key=lambda x: int(x.split('.')[0]))
    ID = [int(imgname.split('.')[0]) for imgname in test_image_names]
    N_samples = len(test_image_names)
    print(f'Found {N_samples} images in test dataset folder: {path.join(*path.split(root_path)[:-1])}'+
          f'\n\"{test_image_names[0]}\"\n\"{test_image_names[1]}\"\n\"{test_image_names[2]}\"\n. . .\n'+
          f'\"{test_image_names[-3]}\"\n\"{test_image_names[-2]}\"\n\"{test_image_names[-1]}\"]\n')
    
    # iter over batches
    pred_labels = []
    N_batches = N_samples//batch_size + (1 if N_samples%batch_size else 0)
    print(f'Processing {N_batches} test batches in total (batch_size={batch_size}).')
    for b in range(N_batches):
        last_idx = min([b*batch_size+batch_size,N_samples])
        # read and transform images
        img_batch = torch.stack([transform( imread(path.join(root_path,imgname)) )
                     for imgname in test_image_names[b*batch_size:last_idx]],dim=0)
        img_batch = img_batch.to(device)
        # predict
        outputs = model(img_batch) # output:(batch, #classes)
        _, preds = torch.max(outputs, 1) # labels:(batch,)
        preds = preds.cpu()
        pred_labels.extend(preds.tolist())
    print('Done.')
    return {'ID': ID, 'Label': pred_labels}

