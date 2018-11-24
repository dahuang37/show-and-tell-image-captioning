import numpy as np
import torch
from base.base_trainer import BaseTrainer
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import time


class Trainer(BaseTrainer):
    """ Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, verbosity, identifier='',
                 valid_data_loader=None, logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, identifier, logger)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader else False

    def _train_epoch(self, epoch):
        """ Train an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            You should modify the data loading part in most cases.
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.train()

        start_time = time.time()
        total_loss = 0
        for batch_idx, (images, captions, lengths, _) in enumerate(self.data_loader):
            
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            self.optimizer.zero_grad()
            output = model(images, captions, lengths)
            loss = self.loss(output, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            # logging info
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, time: {}'.format(
                    epoch, batch_idx * len(images), len(self.data_loader) * len(images),
                    100.0 * batch_idx / len(self.data_loader), loss.item(), format_time(time.time()-start_time)))

        avg_loss = total_loss / len(self.data_loader)
        log = {'loss': avg_loss}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """ Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            Modify this part if you need to.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.eval()
        
        total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (images, captions, lengths, _) in enumerate(self.valid_data_loader):

                images, captions = images.to(device), captions.to(device)

                output = model(images, captions, lengths)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                loss = self.loss(output, targets)
                total_val_loss += loss.item()
                progress_bar(batch_idx, len(self.valid_data_loader))

                
        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        return {'val_loss': avg_val_loss}
