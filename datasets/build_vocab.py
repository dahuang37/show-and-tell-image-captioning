import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import json

class Vocabulary(object):
    """
    Simple vocabulary wrapper.

    Params:
        word2idx: map word to idx 
        idx2word: map idx to word
        idx: number of words
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """ add word to dictionary """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """ return idx or when word not in dict return <unk> """
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold=5):
    """
    build vocab from caption list

    Args:
        json: caption file
        threshold: min number of word count to include words
    """
    coco = COCO(json)
    ids = coco.anns.keys()
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        print(i)
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

# def flickr8k_build_vocab(, threshold=5):
#     ann = json.load(open(json_dict, 'r'))
#     assert type(ann) == dict, 'annotation file format {} not supported'.format(type(ann))
#     ids = ann.keys()
#     counter = Counter()
#     for i, id in enumerate(ids):
#         caption = str(ann[id]['caption'])
#         tokens = nltk.tokenize.word_tokenize(caption.lower())
#         counter.update(tokens)
#     words = [word for word, cnt in counter.items() if cnt >= threshold]
#     vocab = Vocabulary()
#     vocab.add_word('<pad>')
#     vocab.add_word('<start>')
#     vocab.add_word('<end>')
#     vocab.add_word('<unk>')
#
#     for i, word in enumerate(words):
#         vocab.add_word(word)
#     vocab_path = '../data/Flickr8k_text/vocab.pkl'
#     with open(vocab_path, 'wb') as f:
#         pickle.dump(vocab, f)
#     print("Total Vocabulary size: {}".format(len(vocab)))

def save_vocab(json, threshold, file_path):
    """
    Build vocab and save to file for training
    """
    vocab = build_vocab(json=json, threshold=threshold)
    vocab_path = file_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total Vocabulary size: {}".format(len(vocab)))
    # print("Saving to {}".format(vocab_path))

def main(args):
    json = args.json
    threshold = args.threshold
    file_path = args.file_path
    save_vocab(json, threshold, file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flickr8k_json = "../data/flickr8k/Flickr8k_text/buildvocab.json"
    flickr8k_vocab_path = '../data/flickr8k/Flickr8k_text/vocab.pkl'
    coco_json = "./data/coco/annotations/captions_train2014.json"
    coco_vocab_path = "./data/coco/vocab.pkl"
    parser.add_argument('--json', type=str, default= flickr8k_json, help="path for annoations")
    parser.add_argument("--file_path", type=str, default= flickr8k_vocab_path, help="path for vocab file")
    parser.add_argument("--threshold", type=int, default=5, help="min num of word counts")
    args = parser.parse_args()
    main(args)



    
