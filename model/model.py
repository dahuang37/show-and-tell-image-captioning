from base.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence


class BaselineModel(BaseModel):
    ''' Arguments are saved in a dictionary so that hyper-parameter can be re-used during testing '''
    def __init__(self, dictionary):
        super(BaselineModel, self).__init__()
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

    def forward(self, images, captions, lengths):
        ''' Extract features from input '''
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
        outputs = self.dropout(hiddens[0])
        outputs = self.decoder_linear(outputs)

        return outputs

    def inference(self, images, states = None):
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.encoder_linear(features))
        # print(features.shape)
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
        # print(len(inference_output))
        inference_output = torch.stack(inference_output,1)

        return inference_output
    

class Encoder(BaseModel):
    '''
    resnet 
    linear 
    bn           
    '''
    def __init__(self, embed_size, cnn_model="resnet18"):
        super(Encoder, self).__init__()
        # same as models.resnet18(pretrained=True)
        resnet = getattr(models, cnn_model)(pretrained=True)
        # remove the last fc layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, x):
        ''' Extract features from input '''
        with torch.no_grad():
            features = self.resnet(x)
        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.linear(features))
        
        return features

class Decoder(BaseModel):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        # concat features and embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        print(packed[0].shape)
        print(packed[1].shape)
        hiddens, _ = self.rnn(packed)
        print(hiddens[0].shape)
        print(hiddens[1].shape)
        outputs = self.linear(hiddens[0])
        return outputs

    def inference(self, features, states = None):
        inference_output = []
        rnn_input = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.rnn(rnn_input, states)
            rnn_outputs = self.decoder_linear(hiddens.squeeze(1))
            prediction = rnn_outputs.max(1)[1]
            inference_output.append(prediction)
            rnn_input = self.embedding(prediction)
            rnn_input = rnn_input.unsqueeze(1)
        inference_output = torch.cat(inference_output,1)

        return inference_output



