import numpy as np
from eval import *
import torch
from base.base_trainer import BaseTrainer
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import time
import torch.nn as nn



class Trainer(BaseTrainer):
    """ Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, vocab, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, verbosity, id, dataset, identifier='',
                 valid_data_loader=None, logger=None):
        super(Trainer, self).__init__(model, loss, vocab, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, id, dataset, identifier, logger)
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
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self,epoch):
        """ Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            Modify this part if you need to.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model
        loss = nn.CrossEntropyLoss()
        test_path = ''
        if self.dataset == "flickr30k":
            test_path = 'data/flickr30k/captions_flickr30k_val.json'
        elif self.dataset == "flickr8k":
            test_path = 'data/flickr8k/Flickr8k_text/captions_flickr8k_val.json'
        elif self.dataset == "mscoco":
            test_path =  'data/coco/annotations/captions_val2014_reserved.json'
        eval_loss, coco_stat, predictions = eval(self.valid_data_loader, model, self.vocab, loss, test_path)

        avg_val_loss = eval_loss / len(self.valid_data_loader)
        result_dict = {'coco_stat': coco_stat}

        id_filename = str(self.id) + '_/'
        id_file_path = self.save_dir + '/' + id_filename + 'metrics/'
        ensure_dir(id_file_path)
        print("Saving result: {} ...".format(id_file_path))
        load_save_result(epoch,result_dict,id_file_path)

                

        return {'val_loss': avg_val_loss}
