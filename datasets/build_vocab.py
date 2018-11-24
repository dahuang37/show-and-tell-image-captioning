import nltk
nltk.download('punkt')
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


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

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def save_vocab(json, threshold, file_path):
    """
    Build vocab and save to file for training
    """
    vocab = build_vocab(json=json, threshold=threshold)
    vocab_path = file_path
    # with open(vocab_path, 'wb') as f:
        # pickle.dump(vocab, f)

    print("Total Vocabulary size: {}".format(len(vocab)))
    # print("Saving to {}".format(vocab_path))

def main(args):
    json = args.json
    threshold = args.threshold
    file_path = args.file_path
    save_vocab(json, threshold, file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="./data/coco/annotations/captions_train2014.json", help="path for annoations")
    parser.add_argument("--file_path", type=str, default="./data/coco/vocab.pkl", help="path for vocab file")
    parser.add_argument("--threshold", type=int, default=5, help="min num of word counts")
    args = parser.parse_args()
    main(args)



    
