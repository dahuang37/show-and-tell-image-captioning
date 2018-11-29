import numpy as np
import torch
from base.base_trainer import BaseTrainer
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import time
import json
from json import encoder
import random
import string
import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from datasets import Vocabulary
import datasets.dataloader as dataloader
import argparse
from torchvision import transforms
import torch.nn as nn
from model.model import BaselineModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def coco_metric(input_sentence, path_anna ,tmp_file=None):


    coco_set = COCO(path_anna)
    imgid_set = coco_set.getImgIds()

    if tmp_file is None:
        encoder.FLOAT_REPR = lambda o: format(o, '.3f')
        random.seed(time.time())
        tmp_file = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

        
        pred_set = [prediction for prediction in input_sentence if prediction['image_id'] in imgid_set]
        print('using %d/%d predictions' % (len(pred_set), len(input_sentence)))

        ensure_dir('cache/')
        with open('cache/' + tmp_file + '.json', 'w') as f:
            json.dump(pred_set, f)


    result = 'cache/' + tmp_file + '.json'
    cocoRes = coco_set.loadRes(result)
    cocoEval = COCOEvalCap(coco_set, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.system('rm ' + 'cache/' + tmp_file + '.json')


    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    return out

def eval(data_loader, model, dictionary, loss_f, test_path,optimizer=None):
    model.eval()

    total_loss = 0
    start_time = time.time()
    num_loss = 0
    image_set = []
    predictions = []
    with torch.no_grad():
        for batch_id, (images, captions, lengths, img_id) in enumerate(data_loader):
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # computing loss
            output = model(images, captions, lengths)
            # output = pack_padded_sequence(output, lengths, batch_first=True)[0]
            loss = loss_f(output, targets)
            total_loss += loss
            num_loss += 1

            inference_output = model.inference(images)
            # beam_search = model.beam_search(images, k=2)

            inference_output = inference_output.cpu().data.numpy()
            sentence_output = []

            """ convert word token to word""" 
            for sentence_id in inference_output:
                sentence = []
                for word_id in sentence_id:
                    word = dictionary.idx2word[word_id]
                    if word == '<start>':
                        continue
                    if word == '<end>':
                        break
                    sentence.append(word)
                sentence_one = ' '.join(sentence)
                sentence_output.append(sentence_one)

            """ create dict of image id and its caption for eval """ 
            for id, sentence in enumerate(sentence_output):
                if img_id[id] in image_set:
                    continue
                else:
                    image_set.append(img_id[id])
                pred = {'image_id': img_id[id], 'caption': sentence}
                predictions.append(pred)
            progress_bar(batch_id, len(data_loader))

    coco_stat = coco_metric(predictions,test_path)
    eval_loss = total_loss/len(data_loader)

    return eval_loss, coco_stat, predictions


def main(args):

    test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    vocab = dataloader.get_vocab(dataset=args.dataset)()
    data_loader = dataloader.get_data_loader(dataset=args.dataset)(mode="test",
                                                                   transform=test_transform,
                                                                   vocab=vocab,
                                                                   batch_size=args.batch_size,
                                                                   shuffle=False,
                                                                   num_workers=0)
    model = BaselineModel(args.embed_size, args.hidden_size, len(vocab), num_layers=1, cnn_model=args.cnn_model).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    loss = nn.CrossEntropyLoss()

    eval_loss, coco_stat, predictions = eval(data_loader, model, vocab, loss,args.test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show and Tell')
    parser.add_argument('-cp', '--checkpoint_path', type=str,
                        help='checkpoint path to be loaded')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='mini-batch size (default: 4)')
    
    parser.add_argument('--dataset', default="mscoco", type=str,
                        help='dataset used [mscoco | flickr8k | flickr30k | sbu | pascal]')

    parser.add_argument('--embed_size', default=512, type=int,
                        help='dimension for word embedding vector')
    parser.add_argument('--hidden_size', default=512, type=int,
                        help='dimension for lstm hidden layer')
    parser.add_argument('--cnn_model', default="resnet152", type=str,
                        help='pretrained cnn model used')

    parser.add_argument('--test_path', default="data/flickr8k/Flickr8k_text/captions_flickr8k_test.json", type=str,
                        help='pretrained cnn model used')

    main(parser.parse_args())
    # coco_metric(None, "YG7FQK")

