# show-and-tell-image-captioning
Repo for reproducing Show and Tell paper.

## set up

```sh
chmod +x ./set_up.sh
# downlaods the data and set up the folders
./set_up.sh
```


# TODO
### Datasets / Dataloader
- [ ] Flickr30k / Flickr8k
- [ ] SBU
- [ ] PASCAL
### Training
- [ ] Write model for training
- [ ] Supervised training process
- [ ] Record the effect of different hyperparameters
- [ ] Visualize the effect of hyperparameters (e.g. Dropout, hidden_size, etc)
### Evaluation
- [ ] Investigate in evaluation metrics (BLEU, METER, Cider), using MSCOCO api


## Basic Usage
```
usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--dataset DATASET]
               [--embed_size EMBED_SIZE] [--hidden_size HIDEEN_SIZE]
               [--cnn_model CNN_MODEL]

               

Show and Tell

optional arguments:
  -h, --help    show this help message and exit
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for model (default: 0.001)
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
  --dataset DATASET
                        dataset loaded into model (default: mscoco) options: [msococ | flickr8k/30k | sbu | pascal]
  --embed_size EMBED_SIZE
                        dimension for word embedding vector (default: 256)
  --hidden_size HIDEEN_SIZE
                        dimension for lstm hidden layer (default: 512)
  --cnn_model CNN_MODEL
                        pretrained cnn model used for encoder (default: resnet18)
```

## Structure
```
├── base/ - abstract base classes
│   ├── base_model.py - abstract base class for models.
│   └── base_trainer.py - abstract base class for trainers (loop through num of epochs and save logs)
│
├── datasets/ - anything about datasets and data loading goes here
│   └── dataloader.py - main class for returning data loader
|   └── build_vocab.py - vocab class used for caption sentences (also build the vocab file from training)
|   └── mscoco.py - datasets class and data loader for mscoco (also split 4k random val as test)
│
├── data/ - default folder for data
│
├── logger/ - for training process logging
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── saved/ - default checkpoint folder
│   └── model.py - default model
│
├── trainer.py - loop through the data loader 
│
└── utils.py

```
