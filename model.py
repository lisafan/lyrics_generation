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
                 n_layers=1, hidden_size=256, melody_len=40,
                 word_embedding_size=128, word_embeddings=None, 
                 use_artist=True, embed_artist=False, num_artists=10,
                 artist_embedding_size=32, use_melody=True):
        
        super(LyricsRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batchsize = batch_size
        self.pad_id = pad_id

        self.input_size = input_size
        if word_embeddings:
            self.word_embed_size = word_embeddings.shape[1]
            self.word_encoder = nn.Embedding.from_pretrained(word_embeddings)
        else:
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

        self.use_melody = use_melody
        self.melody_len = melody_len
        if self.use_melody:
            self.lstm_input_size += self.melody_len
            
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, output_size)
        
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
        return self.lyrics_generator(input,input_lens)
        

    def lyrics_generator(self,input,input_lens):
        if self.use_artist:
            if self.use_melody:
                input,melody_input,artist_input = input
            else:
                input,artist_input = input
        else:
            if self.use_melody:
                input,melody_input = input
        
        embed = self.word_encoder(input)
        
        if self.use_artist:
            # repeat artist along sequence
            artist_input = torch.unsqueeze(artist_input,dim=1)
            artist_input = artist_input.expand(-1,input.size()[1]).to(device)
            
            artist_embed = self.artist_encoder(artist_input)
            
            # concatenate artist embedding to word embeddings
            embed = torch.cat([embed, artist_embed],dim=2)

        if self.use_melody:
            melody_input = melody_input.view(-1, 1, self.melody_len)
            melody_input = melody_input.repeat(1, embed.shape[1], 1)
            embed = torch.cat([embed, melody_input],dim=2)
            
            # sort by numwords in reverse order
            input_len_order = np.argsort(input_lens)[::-1]
            emb = torch.zeros_like(embed)
            sorted_input_lens = []
            for i in range(len(input_lens)):
                emb[i] = embed[input_len_order[i]]
                sorted_input_lens += [input_lens[input_len_order[i]]]

            emb_pad = rnn.pack_padded_sequence(emb, sorted_input_lens, batch_first=True)
            out_pad, self.hidden = self.lstm(emb_pad, self.hidden)
            out, _ = rnn.pad_packed_sequence(out_pad, batch_first=True)

            # put back in [batch x line] order
            output = torch.zeros_like(out)
            for i in range(len(input_lens)):
                output[input_len_order[i]] = out[i]
        else:
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

    def evaluate_seq(self, prime_str, artist=None, melody=None, predict_line_len=5, predict_seq_len=20, temperature=0.8):
        self.eval()
        with torch.no_grad():
            self.hidden = self.init_hidden()

            # repeat input across batches
            if self.use_artist:
                artist = torch.from_numpy(np.array([artist]*self.batchsize))
            if self.use_melody:
                melody = melody.repeat(self.batchsize, 1, 1)

            # repeat lines across batch
            inp = torch.LongTensor([l[0] for l in prime_str]*self.batchsize)
            inp = torch.unsqueeze(inp,dim=1).to(device)

            input_lens = [1] * (self.batchsize*predict_line_len)
            predicted = [[l[0]] for l in prime_str]

            for i in range(predict_seq_len):
                # just get first batch (num lines x vocab dist)
                output = self.lyrics_generator([inp, melody], input_lens)[0]

                # for each line
                for j in range(predict_line_len):
                    # if there's more of prime string, use it
                    if i < len(prime_str[j])-1:
                        top_i = prime_str[j][i+1]
                    else:
                        # Sample from the network as a multinomial distribution
                        output_dist = output[j].data.view(-1).div(temperature).exp()
                        top_i = torch.multinomial(output_dist, 1)[0]

                    # Add predicted character to string and use as next input
                    predicted[j] += [top_i]

                inp = torch.LongTensor([l[-1] for l in predicted]*self.batchsize)
                inp = torch.unsqueeze(inp,dim=1).to(device)

            self.train()
            return predicted
    
    # def evaluate(self, prime_str=[START], artist=None, predict_len=100, temperature=0.8):
    def evaluate(self, prime_str, artist=None, melody=None, predict_len=100, temperature=0.8):
        self.hidden = self.init_hidden()
        
        # repeat input across batches
        prime_input = torch.LongTensor(prime_str).expand(self.batchsize,-1).to(device)
        predicted = prime_str
        input_lens = [len(prime_str)-1]*self.batchsize
        if self.use_artist:
            artist = torch.from_numpy(np.array([artist]*self.batchsize))
        if self.use_melody:
            melody = melody.repeat(self.batchsize, 1)
            
        def get_input(inp):
            if self.use_artist:
                if self.use_melody:
                    return [inp, melody, artist]
                else:
                    return [inp, artist]
            else:
                if self.use_melody:
                    return [inp, melody]
                else:
                    return inp

        if len(prime_str) > 1:
            # Use priming string to "build up" hidden state
            self.lyrics_generator(get_input(prime_input[:,:-1]), input_lens)
            
        inp = prime_input[:,-1].view(self.batchsize,1).to(device)
        input_lens = [1]*self.batchsize
        
        for p in range(predict_len):
            # just get first row, since all rows are the same
            output = self.lyrics_generator(get_input(inp), input_lens)[0]

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted += [top_i]
            inp = torch.LongTensor([top_i]).expand(self.batchsize,1).to(device)

        return predicted
