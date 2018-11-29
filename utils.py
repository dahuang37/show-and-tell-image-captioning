import os
import sys
import time
import math
import numpy as np
import pickle
#import matplotlib
#import matplotlib.pyplot as plt
import json
from pathlib import Path
import functools

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 60.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    # running avg step size
    
    # est_time = max((tot_time/i),step_time)*total 

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    # L.append(' | Est: %s' % format_time(est_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class Early_stopping(object):
    '''
    monitor on val loss
    if stop improving from the past 10 epoch, then stop
    '''
    def __init__(self, in_use, patience=15):
        self.in_use = in_use
        self.patience = patience
        self.wait = 0
        self.current_best = np.inf
        self.stopping = False

    def update(self, test_loss):
        if not self.in_use:
            return
        if self.current_best > test_loss:
            self.current_best = test_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopping = True

    def stop(self):
        return self.stopping


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_save_hyper(args, id_to_hyper_filename= "/id_to_hyper.json"):
    args.save_dir = os.path.join(args.save_dir, args.dataset)
    ensure_dir(args.save_dir)
    # find ids for current hyper
    hyper_id = 0
    id_to_hyper_path = args.save_dir + id_to_hyper_filename
    id_to_hyper_file = Path(id_to_hyper_path)
    id_to_hyper_dict = {}

    #id_list = [int(dI.split("_")[0]) for dI in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, dI))]
    #if len(id_list) != 0:

    # load hyper dict
    if not id_to_hyper_file.exists():
        id_to_hyper_dict[hyper_id] = vars(args)
    else:
        with open(id_to_hyper_path, "r") as f:
            id_to_hyper_dict = json.load(f)

        hyper_id = max(map(int,id_to_hyper_dict.keys())) + 1
        id_to_hyper_dict[hyper_id] = vars(args)
    # save json
    with open(id_to_hyper_path, "w") as f:
        json.dump(id_to_hyper_dict, f)

    return hyper_id

def load_save_result(epoch,mode,data,filepath, filename= "/results.json"):
    if mode == 'val':
        result_json = []
        result_path = filepath + filename
        result_file = Path(result_path)

        # load hyper dict
        if not result_file.exists():
            result_json.append(data)
        else:
            with open(result_path, "r") as f:
                result_json = json.load(f)
            result_json.append(data)
        # save json
        with open(result_path, "w") as f:
            json.dump(result_json, f)

    elif mode == 'test':
        result_json = {}
        result_path = filepath + filename
        result_file = Path(result_path)
        # load hyper dict
        if not result_file.exists():
            result_json[epoch] = data
        else:
            with open(result_path, "r") as f:
                result_json = json.load(f)
            result_json[epoch] = data
        # save json
        with open(result_path, "w") as f:
            json.dump(result_json, f)









