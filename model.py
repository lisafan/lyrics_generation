#!/usr/bin/env python
# coding: utf-8

# # Pre-processing data
# Load lyrics with artist info


import sys,os
import string, re
import unidecode
import random, math, time
import pickle
import numpy as np
import torch
import tensorboardX
import matplotlib as plt
from collections import Counter
from torch import nn
from torch.nn.utils import rnn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LyricsRNN(nn.Module):
    def __init__(self, input_size, output_size, pad_id, batch_size=8, 
                 n_layers=1, hidden_size=256, word_embedding_size=128, 
                 use_artist=True, embed_artist=False, num_artists=10, artist_embedding_size=32):
        
        super(LyricsRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batchsize = batch_size
        self.pad_id = pad_id

        self.input_size = input_size
        self.word_embed_size = word_embedding_size
        self.word_encoder = nn.Embedding(self.input_size, self.word_embed_size,padding_idx=self.pad_id)
        self.lstm_input_size = self.word_embed_size
        
        self.use_artist = use_artist
        if self.use_artist:
            self.num_artists = num_artists
            # either embed artist data or use a one-hot vector
            if embed_artist:
                self.artist_embed_size = artist_embedding_size
                self.artist_encoder = nn.Embedding(self.num_artists, self.artist_embed_size)
            else:
                self.artist_embed_size = self.num_artists
                self.artist_encoder = self.artist_onehot

                    
            self.lstm_input_size += self.artist_embed_size
            
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, output_size)
        print(output_size)
        
        self.hidden = self.init_hidden()
    
    def artist_onehot(self, artist):
        tensor = torch.zeros(self.batchsize, artist.size()[1], self.artist_embed_size).to(device)
        for i in range(self.batchsize): #tensor.size()[0]):
            idx = artist[i,0]
            tensor[i,:,idx] = 1
        return tensor
    
    def init_hidden(self):
         (Variable(torch.randn(self.n_layers, self.batchsize, self.hidden_size)).to(device),
                Variable(torch.randn(self.n_layers, self.batchsize, self.hidden_size)).to(device))

    def forward(self, input, input_lens):
        self.hidden = self.init_hidden()
        
        if self.use_artist:
            input,artist_input = input
        
        embed = self.word_encoder(input)
        
        if self.use_artist:
            # repeat artist along sequence
            artist_input = torch.unsqueeze(artist_input,dim=1)
            artist_input = artist_input.expand(-1,input.size()[1]).to(device)
            
            artist_embed = self.artist_encoder(artist_input)
            
            # concatenate artist embedding to word embeddings
            embed = torch.cat([embed, artist_embed],dim=2)

        emb_pad = rnn.pack_padded_sequence(embed, input_lens, batch_first=True)
        out_pad, self.hidden = self.lstm(emb_pad, self.hidden)
        output, _ = rnn.pad_packed_sequence(out_pad, batch_first=True)
        
        # second RNN goes here

        output = self.dropout(output)

        output = output.contiguous().view(-1,output.shape[2])
        output = self.linear(output)
        output = F.log_softmax(output,dim=1)
        output = output.view(self.batchsize, -1, self.output_size)
        
        return output

    # cross entropy loss with padding
    def loss(self, Y_hat, Y):
        Y = Y.view(-1)
        Y_hat = Y_hat.view(-1,self.output_size)
        mask = (Y != self.pad_id).float()

        non_pad_tokens = torch.sum(mask).item()
        # gets index of correct word in vocab, masks to 0 if padding
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        
        loss = -torch.sum(Y_hat) / non_pad_tokens
        return loss
    
    # def evaluate(self, prime_str=[START], artist=None, predict_len=100, temperature=0.8):
    def evaluate(self, prime_str, artist=None, predict_len=100, temperature=0.8):
        self.hidden = self.init_hidden()
        
        # repeat input across batches
        prime_input = torch.LongTensor(prime_str).expand(self.batchsize,-1).to(device)
        predicted = prime_str
        input_lens = [len(prime_str)-1]*self.batchsize
        if self.use_artist:
            artist = torch.from_numpy(np.array([artist]*self.batchsize))
            
        def get_input(inp):
            if self.use_artist:
                return [inp, artist]
            else:
                return inp

        if len(prime_str) > 1:
            # Use priming string to "build up" hidden state
            self.forward(get_input(prime_input[:,:-1]), input_lens)
            
        inp = prime_input[:,-1].view(self.batchsize,1).to(device)
        input_lens = [1]*self.batchsize
        
        for p in range(predict_len):
            # just get first row, since all rows are the same
            output = self.forward(get_input(inp), input_lens)[0]

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted += [top_i]
            inp = torch.LongTensor([top_i]).expand(self.batchsize,1).to(device)

        return predicted
