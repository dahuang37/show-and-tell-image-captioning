import argparse
import torch.optim as optim
from model.model import Model
# from model.loss import my_loss
# from model.metric import my_metric, my_metric2
from datasets import Vocabulary
import datasets.dataloader as dataloader
from trainer import Trainer
from logger.logger import Logger
import pickle
from torchvision import transforms
from utils import *


parser = argparse.ArgumentParser(description='Show and Tell')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=32, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='model/saved', type=str,
                    help='directory of saved model (default: model/saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
# parser.add_argument('--data-dir', default='datasets', type=str,
#                     help='directory of training/testing data (default: datasets)')


def main(args):
    # Model
    model = Model()
    model.summary()

    # A logger to store training process information
    logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    # loss = my_loss
    # metrics = [my_metric, my_metric2]
    optimizer = optim.Adam(model.parameters())

    # Data loader and validation split
    data_loader = dataloader.get_data_loader(dataset="mscoco")(mode="train",
                                                               transform=transforms.ToTensor(),
                                                               batch_size=args.batch_size,
                                                               num_workers=4)
    valid_data_loader = dataloader.get_data_loader(dataset="mscoco")(mode="val",
                                                                     transform=transforms.ToTensor(),
                                                                     batch_size=args.batch_size,
                                                                     num_workers=4)
    # An identifier (prefix) for saved model
    identifier = type(model).__name__ + '_'

    # Trainer instance
    trainer = Trainer(model, None, None,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      identifier=identifier,
                      )

    # # Start training!
    trainer.train()

    # # See training history
    # print(logger)


if __name__ == '__main__':
    main(parser.parse_args())
