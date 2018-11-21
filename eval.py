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
from torch.autograd import Variable
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_metric(input_sentence):
    sys.path.append("coco-caption")
    path_anna = "data/coco/annotations/captions_val2014.json"
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    random.seed(time.time())
    tmp_file = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

    coco_set = COCO(path_anna)
    imgid_set = coco_set.getImgIds()

    pred_set = [prediction for prediction in input_sentence if prediction['image_id'] in imgid_set]
    print('using %d/%d predictions' % (len(pred_set), len(input_sentence)))

    with open('cache/' + tmp_file + '.json', 'w') as f:
        json.dump(pred_set, f)

    json.dump(pred_set, open('cache/' + tmp_file + '.json', 'w'))

    result = 'cache/' + tmp_file + '.json'
    cocoRes = coco_set.loadRes(result)
    cocoEval = COCOEvalCap(coco_set, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    #keep output to read first
    #os.system('rm ' + 'cache/' + tmp_file + '.json')


    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def eval(data_loader, model, dictionary, loss_f, optimizer):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    start_time = time.time()
    num_loss = 0
    image_set = []
    predictions = []


    for batch_id, (images, captions, lengths, img_id) in enumerate(data_loader):
        images, captions = images.to(device), captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        optimizer.zero_grad()
        output = model(images, captions, lengths)
        loss = loss_f(output, targets)
        total_loss += loss
        num_loss += 1

        inference_output = model.inference(images)

        inference_output = inference_output.cpu().data.numpy()
        sentence_output = []
        for sentence_id in inference_output:
            sentence = []
            for word_id in sentence_id:
                word = dictionary.idx2word[word_id]
                if word == '<end>':
                    break
                sentence.append(word)
            sentence_one = ''.join(sentence)
            sentence_output.append(sentence_one)
        for id, sentence in enumerate(sentence_output):
            if img_id[id] in image_set:
                continue
            else:
                image_set.append(img_id[id])
            pred = {'img_id': img_id[id], 'caption': sentence}
            predictions.append(pred)

    coco_stat = coco_metric(predictions)
    eval_loss = total_loss/num_loss
    return eval_loss, coco_stat, predictions




