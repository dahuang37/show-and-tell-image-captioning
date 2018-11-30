from base.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineModel(BaseModel):
    ''' Arguments are saved in a dictionary so that hyper-parameter can be re-used during testing '''
    def __init__(self, dictionary):
        super(BaselineModel, self).__init__()
        self.vocab_size = dictionary['vocab_size']
        self.embed_size = dictionary['embed_size']

        resnet = getattr(models, dictionary['cnn_model'])(pretrained=True)
        # remove the last fc layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.encoder_linear = nn.Linear(resnet.fc.in_features, dictionary['embed_size'])
        self.bn = nn.BatchNorm1d(dictionary['embed_size'])

        self.embedding = nn.Embedding(dictionary['vocab_size'],dictionary['embed_size'])
        self.rnn = getattr(nn, dictionary['rnn_model'])(dictionary['embed_size'], dictionary['hidden_size'], dictionary['num_layers'], batch_first=True)
        self.dropout = nn.Dropout(dictionary['dropout'])
        self.decoder_linear = nn.Linear(dictionary['hidden_size'], dictionary['vocab_size'])
        
        self.init_weights()

    def init_weights(self):
        init_value = 0.1
        self.embedding.weight.data.uniform_(-init_value, init_value)
        self.encoder_linear.weight.data.uniform_(-init_value, init_value)
        self.decoder_linear.weight.data.uniform_(-init_value,init_value)

    def forward(self, images, captions, lengths):
        ''' Extract features from input and pass output to LSTM'''
        ### ENCODER
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.encoder_linear(features))

        ### DECODER
        embeddings = self.embedding(captions)
        # concat features and embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        hiddens, _ = self.rnn(packed)
        # outputs = self.dropout(hiddens[0])
        outputs = self.decoder_linear(outputs)

        return outputs

    def inference(self, images, states=None):
        # get encoding from cnn
        with torch.no_grad():
            features = self.resnet(images)

        # transform into embed_size
        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.encoder_linear(features))

        inference_output = None
        inference_output = []
        rnn_input = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.rnn(rnn_input, states)
            rnn_outputs = self.decoder_linear(hiddens.squeeze(1))
            prediction = rnn_outputs.max(1)[1]

            inference_output.append(prediction)
            rnn_input = self.embedding(prediction)
            rnn_input = rnn_input.unsqueeze(1)
        inference_output = torch.stack(inference_output,1)

        return inference_output

    def beam_search(self, images, vocab, states=None, k=20, max_length=20):
        embed_size = self.embed_size
        vocab_size = self.vocab_size
        batch_size = images.shape[0]

        # Encode
        with torch.no_grad():
            features = self.resnet(images)
    
        # Flatten encoding
        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.encoder_linear(features))
        # We'll treat the problem as having a batch size of k
        encoder_out = features.expand(k, embed_size)  # (k, encoder_dim)

        # # Tensor to store top k previous words at each step; now they're just <start>
        # k_prev_words = None #encoder_out  # (k, 1)
        
        # # Tensor to store top k sequences; now they're just <start>
        # seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        embeddings = encoder_out.unsqueeze(1)
        h, c = None, None
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            # embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            # hiddens is h_t, which is one in this case
            if step == 1:
                hiddens, (h, c) = self.rnn(embeddings, states)  # (s, decoder_dim)
            else:
                hiddens, (h, c) = self.rnn(embeddings, (h,c)) 
            scores = self.decoder_linear(hiddens.squeeze(1))
            scores = F.log_softmax(scores, dim=1)
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            if step == 1:
                seqs = next_word_inds.unsqueeze(1)
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab('<end>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            # h
            h = h[:,prev_word_inds[incomplete_inds]]
            # c
            c = c[:,prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            embeddings = self.embedding(k_prev_words)  # (s, embed_dim)
            
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        return seq
        
