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

class SemanticLyricsRNN(nn.Module):
    def __init__(self, input_size, output_size, pad_id, batch_size=8, 
                 noise_size=32, n_layers_S=1, hidden_size_S=128, n_layers_L=1, hidden_size_L=256,
                 melody_len = 40, word_embedding_size=128, word_embeddings=None, 
                 use_artist=True, embed_artist=False, num_artists=10, artist_embedding_size=32,
                 use_noise=False, use_melody=True):
        
        super(SemanticLyricsRNN, self).__init__()
        self.hidden_size_S = hidden_size_S
        self.hidden_size_L = hidden_size_L
        self.n_layers_S = n_layers_S
        self.n_layers_L = n_layers_L
        self.batchsize = batch_size
        self.output_size = output_size
        self.pad_id = pad_id

        self.input_size = input_size
        self.noise_size = noise_size
        self.sem_input_size = self.noise_size
        self.use_noise = use_noise

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
                    
            self.sem_input_size += self.artist_embed_size

        self.use_melody = use_melody
        self.melody_len = melody_len
            
        if word_embeddings:
            self.word_embed_size = word_embeddings.shape[1]
            self.word_encoder = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.word_embed_size = word_embedding_size
            self.word_encoder = nn.Embedding(self.input_size, self.word_embed_size,padding_idx=self.pad_id)

        self.lyr_input_size = self.word_embed_size + self.hidden_size_S
        if self.use_melody:
            self.lyr_input_size += self.melody_len
        
        self.semantic_lstm = nn.LSTM(self.sem_input_size, hidden_size_S, n_layers_S, batch_first=True, dropout=0.5)
        self.lyrics_lstm = nn.LSTM(self.lyr_input_size, hidden_size_L, n_layers_L, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size_L, output_size)
        
        self.hidden_S = self.init_hidden_S()
        self.hidden_L = self.init_hidden_L()
    
    def artist_onehot(self, artist):
        tensor = torch.zeros(self.batchsize, artist.size()[1], self.artist_embed_size).to(device)
        for i in range(self.batchsize): 
            idx = artist[i,0]
            tensor[i,:,idx] = 1
        return tensor
    
    def init_hidden_S(self):
         (Variable(torch.randn(self.n_layers_S, self.batchsize, self.hidden_size_S)).to(device),
                Variable(torch.randn(self.n_layers_S, self.batchsize, self.hidden_size_S)).to(device))

    def init_hidden_L(self):
         (Variable(torch.randn(self.n_layers_L, self.batchsize, self.hidden_size_L)).to(device),
                Variable(torch.randn(self.n_layers_L, self.batchsize, self.hidden_size_L)).to(device))

    def get_iteration_noise(self, num_lines):
        return Variable(torch.FloatTensor(self.batchsize, num_lines, self.noise_size).normal_()).to(device)

    # input: (batchsize * numlines) x numwords
    def forward(self, input, input_lens):

        if self.use_artist:
            if self.use_melody:
                input,melody_input,artist_input = input
            else:
                input,artist_input = input
                melody_input = None
        else:
            if self.use_melody:
                input,melody_input = input
            else:
                artist_input = None
                melody_input = None
        num_lines = int(input.size()[0] / self.batchsize)

        self.hidden_S = self.init_hidden_S()
        self.hidden_L = self.init_hidden_L()

        # only used for the "no-semantics" model version to check semantic representations can be learned
        if self.use_noise:
            sem_reps = Variable(torch.FloatTensor(self.batchsize, num_lines, self.hidden_size_S).normal_()).to(device)
        else:
            sem_reps = self.semantic_generator(artist_input, num_lines)

        sem_reps = sem_reps.contiguous().view(-1, self.hidden_size_S)       # (batchsize * numlines) x hiddensize S
        sem_reps = torch.unsqueeze(sem_reps,dim=1)
        sem_reps = sem_reps.expand(-1, input.size()[1], self.hidden_size_S) # (batchsize * numlines) x numwords x hiddensize S

        output = self.lyrics_generator(input, input_lens, sem_reps, melody_input)
        
        return output

    def semantic_generator(self, artists, num_lines):
        # may want to use zeros instead of random noise?
        sem_input = self.get_iteration_noise(num_lines) # batchsize x numlines x noise

        if self.use_artist:        
            # repeat artist along sequence
            artist_input = torch.unsqueeze(artists,dim=1)
            artist_input = artist_input.expand(-1,num_lines).to(device)
            
            artist_embed = self.artist_encoder(artist_input)
            
            # concatenate artist embedding to random noise
            sem_input = torch.cat([sem_input, artist_embed],dim=2)  # batchsize x numlines x (noise + artist embed size)

        sem_reps, _ = self.semantic_lstm(sem_input, self.hidden_S)  # batchsize x numlines x hiddensize S

        return sem_reps

    def lyrics_generator(self, input, input_lens, sem_reps, melody):
        # get word embeddings and concatenate context vectors
        embed = self.word_encoder(input)               # (batchsize * numlines) x numwords x word embed size
        lyr_inp = torch.cat([embed, sem_reps], dim=2)  # (batchsize * numlines) x numwords x (word embed size + hiddensize S)

        if self.use_melody:
            melody = melody.view(-1,1,self.melody_len)
            melody = melody.repeat(1, lyr_inp.shape[1], 1)
            lyr_inp = torch.cat([lyr_inp, melody], dim=2)  # (batchsize * numlines) x numwords x (word embed size + hiddensize S + melody size)

        # sort by numwords in reverse order
        input_len_order = np.argsort(input_lens)[::-1]
        lyr_input = torch.zeros_like(lyr_inp)
        sorted_input_lens = []
        for i in range(len(input_lens)):
            lyr_input[i] = lyr_inp[input_len_order[i]]
            sorted_input_lens += [input_lens[input_len_order[i]]]

        inp_pad = rnn.pack_padded_sequence(lyr_input, sorted_input_lens, batch_first=True)
        out_pad, self.hidden = self.lyrics_lstm(inp_pad, self.hidden_L)
        out, _ = rnn.pad_packed_sequence(out_pad, batch_first=True)

        # put back in [batch x line] order
        output = torch.zeros_like(out)
        for i in range(len(input_lens)):
            output[input_len_order[i]] = out[i]

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
    
    def evaluate(self, prime_str, artist=None, melody=None, predict_line_len=5, predict_seq_len=20, temperature=0.8):
        self.eval()
        with torch.no_grad():
            self.hidden_S = self.init_hidden_S()
            self.hidden_L = self.init_hidden_L()

            # repeat input across batches
            if self.use_artist:
                artist = torch.from_numpy(np.array([artist]*self.batchsize))
            if self.use_melody:
                melody = melody.repeat(self.batchsize, 1, 1)
            
            # get semantic rep for each line and reshape
            sem_reps = self.semantic_generator(artist, predict_line_len)
            sem_reps = sem_reps.contiguous().view(-1, self.hidden_size_S)       # (batchsize * numlines) x hiddensize S
            sem_reps = torch.unsqueeze(sem_reps,dim=1)                          # (batchsize * numlines) x 1 x hiddensize S

            # repeat lines across batch
            inp = torch.LongTensor([l[0] for l in prime_str]*self.batchsize)
            inp = torch.unsqueeze(inp,dim=1).to(device)                         # (batchsize * numlines) x 1 x 1

            input_lens = [1] * (self.batchsize*predict_line_len)
            predicted = [[l[0]] for l in prime_str]

            for i in range(predict_seq_len):
                # just get first batch (num lines x vocab dist)
                output = self.lyrics_generator(inp, input_lens, sem_reps, melody)[0]

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
