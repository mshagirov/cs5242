---
# CS5242
---
> CS 5242 Final Project Code

*Author: Murat Shagirov*.

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

## Train and predict test set labels
Run provided `run.py` with two arguments (path to training data, and path to test data), e.g.:
```
python run.py ../datasets/train_data/ ../datasets/test_data/
```
output should look like this:
```
Training data path:../datasets/train_data/
Test data path:../datasets/test_data/

device: cuda
dtype: torch.float32
Training dataset: 931 samples. 
Validation dataset: 233 samples.

Downloading pre-trained denseNet121 ...

Training the model:
Epoch 0/24 --- train Loss: 0.8766 Acc: 0.6208 || val Loss: 0.4954 Acc: 0.7597 || 43s
Epoch 1/24 --- train Loss: 0.4981 Acc: 0.8013 || val Loss: 0.2577 Acc: 0.8798 || 89s
...
Epoch 24/24 --- train Loss: 0.0572 Acc: 0.9839 || val Loss: 0.1058 Acc: 0.9571 || 1085s
Training complete in 18m 5s
Best val Acc: 0.965665 (return best:True)

Model Evaluation
----------
Loading: train
Loading: val
Losses: {'train': 0.012567672101764049, 'val': 0.10995280875984603}
Accuracies: {'train': tensor(0.9968), 'val': tensor(0.9657)}
~~~~~~~~~~

Found 292 images in test dataset folder: ../datasets/test_data
"0.png"
"1.png"
"2.png"
. . .
"289.png"
"290.png"
"291.png"]

Processing 73 test batches in total (batch_size=4).
Done.
Saving test_result.csv
```

## Files
```
./
    README.md
    run.py
    ckp/
    code/
        0_dataset.ipynb
        1_training_densenet.ipynb
        2_prediction.ipynb
        requirements.txt
        datautils.py
        nn.py
        prediction.py
        train_densenet.py
```

## Notebooks
Run notebooks in oder (0, 1, 2) to explore the dataset, to finetune pre-trained models, and to predict test set results. Alternatively, edit provided scripts to train and predict test set labels in `./code/`.

