# show-and-tell-image-captioning
Repo for reproducing Show and Tell paper.

## set up

```sh
chmod +x ./set_up.sh
# downlaods the data and set up the folders
./set_up.sh
```

<!-- ## Basic Usage
The code in this repo is an MNIST example of the template, try run:
```
python main.py
```
The default arguments list is shown below:
```
usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--data-dir DATA_DIR]
               [--validation-split VALIDATION_SPLIT] [--no-cuda]

PyTorch Template

optional arguments:
  -h, --help    show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs (default: 32)
  --resume RESUME
                        path to latest checkpoint (default: none)
  --verbosity VERBOSITY
                        verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)
  --save-dir SAVE_DIR
                        directory of saved model (default: model/saved)
  --save-freq SAVE_FREQ
                        training checkpoint frequency (default: 1)
  --data-dir DATA_DIR
                        directory of training/testing data (default: datasets)
  --validation-split VALIDATION_SPLIT
                        ratio of split validation data, [0.0, 1.0) (default: 0.0)
  --no-cuda   use CPU in case there's no GPU support
```
You can add your own arguments. -->
<!-- 
## Structure
```
├── base/ - abstract base classes
│   ├── base_data_loader.py - abstract base class for data loaders.
│   ├── base_model.py - abstract base class for models.
│   └── base_trainer.py - abstract base class for trainers
│
├── data_loader/ - anything about data loading goes here
│   └── data_loader.py
│
├── datasets/ - default dataset folder
│
├── logger/ - for training process logging
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── modules/ - submodules of your model
│   ├── saved/ - default checkpoint folder
│   ├── loss.py
│   ├── metric.py
│   └── model.py
│
├── trainer/ - trainers for your project
│   └── trainer.py
│
└── utils
     ├── utils.py
     └── ...

```

 -->