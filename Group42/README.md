---
# cs5242
---
> GROUP 42: CS 5242 Final Project Code

*Author: Murat Shagirov*

## Files
```
./Group42/
    README.md
    0_dataset.ipynb
    1_training_densenet.ipynb
    2_prediction.ipynb
    datautils.py
    nn.py
    prediction.py
    train_densenet.py
```
## Requirements
### Python version
- python3.7

### Packages
- jupyter (optional)
- numpy==1.18.1
- pytorch==1.7.0
- torchvision==0.8.1
- pandas==1.0.3
- skimage==0.17.2
- matplotlib==3.2.0

## Notebooks
You can run notebooks in oder (0, 1, 2) to explore the dataset, training models, and prediction. Alternatively, edit provided scripts to train and predict test set labels (see below).

## Training pre-trained (ImageNet) models
First edit "settings" section of the `train_densenet.py`, then run it:
```
> python train_densenet.py
```
Example output (for submission need >60 epochs):
```
device: cuda
dtype: torch.float32
Training dataset: 931 samples. 
Validation dataset: 233 samples.
Epoch 0/2 --- train Loss: 0.7287 Acc: 0.6960 || val Loss: 0.2939 Acc: 0.8755 || 45s
Epoch 1/2 --- train Loss: 0.5300 Acc: 0.7884 || val Loss: 0.3333 Acc: 0.8670 || 89s
Epoch 2/2 --- train Loss: 0.4485 Acc: 0.8260 || val Loss: 0.4894 Acc: 0.8240 || 134s
Training complete in 2m 14s
Best val Acc: 0.875536 (return best:True)
```
after training models are saved to your desired folder.

## Prediction using saved models
Edit and include your model file name and path in `prediction.py`, then run it
```
> python prediction.py
```
Example output:
```
device: cuda
dtype: torch.float32
Training dataset: 931 samples. 
Validation dataset: 233 samples.
Loading: train
Loading: val
Losses: {'train': 0.2859504673956112, 'val': 0.2939242842835034}
Accuracies: {'train': tensor(0.8647), 'val': tensor(0.8755)}
Found 292 images in test dataset folder: ./datasets/test_image/test_image
"0.png"
"1.png"
"2.png"
. . .
"289.png"
"290.png"
"291.png"]

Processing 73 test batches in total (batch_size=4).
Done.
```
this script will produce a "`*.csv`" file for submission.

