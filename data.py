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
import matplotlib.pyplot as plt
import itertools
from collections import Counter, defaultdict
from torch import nn
from torch.nn.utils import rnn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PAD_ID = 0

class LyricsDataset(Dataset):
    def __init__(self, pkl_file, vocab_file=None, vocab_size=10000, chunk_size=0, max_line_len=5, max_seq_len=50, max_mel_len=40, use_semantics=True, use_artist=True, use_melody=True):
        """
        Args:
            csv_file (string): Path to the csv file with lyrics.
            chunk_size (int): Number of lyric lines to use as single sample. If 0, use song's entire lyrics
        """

        # special tokens
        self.EOL = '<EOL>'
        self.UNK = '<UNK>'
        self.START = '<START>'
        self.END = '<END>'
        self.PAD = '<padding>'
        self.PAD_ID = PAD_ID

        self.lyrics = pickle.load(open(pkl_file,'rb'), encoding='latin1')
        
        if vocab_file == None:
            vocab_file = re.sub('.pkl','.vocab',pkl_file)
            if not os.path.exists(vocab_file):
                self.create_vocab(vocab_file)
        
        print("Using vocab file: %s"%vocab_file)            
        self.vocab = [x.split()[0] for x in open(vocab_file).read().splitlines()][:vocab_size]
        self.vocab = [self.START, self.END, self.EOL, self.UNK] + self.vocab
        self.vocab.insert(self.PAD_ID, self.PAD)
        self.vocab_len = len(self.vocab)
        self.max_seq_len = max_seq_len
        self.max_mel_len = max_mel_len
        
        self.use_semantics = use_semantics
        self.use_melody = use_melody
        if self.use_semantics or self.use_melody:
            assert(chunk_size==0 or max_line_len <= chunk_size)
            self.max_line_len = max_line_len

        self.use_artist = use_artist
        if self.use_artist:
            self.artists = sorted(set([x['artist'] for x in self.lyrics]))
            self.num_artists = len(self.artists)
        else:
            self.num_artists = 0
            
        # chunk lyrics
        print("chunking lyrics")
        self.chunk_size = chunk_size
        if self.chunk_size > 0:
            chunked_lyrics = []
            for song in self.lyrics:
                lines = re.split(r'\n',song['lyrics'])
                if self.use_melody:
                    melody = song['melody']
                for i in range(len(lines) - self.chunk_size+1):
                    chunk = '\n'.join(lines[i:i+self.chunk_size])
                    song['lyrics'] = chunk
                    song['melody'] = melody[i:i+self.chunk_size]
                    chunked_lyrics += [song.copy()]
            self.lyrics = chunked_lyrics
                    
    def create_vocab(self,file_name):
        num_songs = len(self.lyrics)
        print('creating vocabulary for %d songs'%num_songs)
        
        vocab = []
        for i,e in enumerate(self.lyrics):
            if i%(num_songs/10)==0:
                print('finished %d/%d songs (%.2f%%)'%(i,num_songs,float(i)/num_songs))
            vocab += [w.lower() for w in e['lyrics'].split()]
        vocab = Counter(vocab)
        
        # save up to 100,000 words
        with open(file_name,'w') as f:
            for i,(a,n) in enumerate(vocab.most_common()):
                if i==100000:
                    break
                if n < 5:
                    break
                f.write('%s\t%s\n'%(a,n))

    def __len__(self):
        # or length of chunked lyrics?
        return len(self.lyrics)

    def __getitem__(self, idx):
        samp = self.lyrics[idx]
        sample = {'inp_words':[],'out_words':[],'inp_ids':[],'out_ids':[],'artist':[],'artist_id':[], 'melody':[], 'song':[]}
        
        #if self.use_semantics:
        if hasattr(self, 'max_line_len'):
            tokenized_lyrics = [[self.START]+line.split()+[self.EOL]  for line in samp['lyrics'].split('\n')]
            tokenized_lines = tokenized_lyrics[:self.max_line_len]

            sample['inp_words'] = [line[:-1][:self.max_seq_len] for line in tokenized_lines]
            sample['out_words'] = [line[1:self.max_seq_len+1] for line in tokenized_lines]
            sample['inp_ids'] = [self.word_tensor(line) for line in sample['inp_words']]
            sample['out_ids'] = [self.word_tensor(line) for line in sample['out_words']]

        else:
            tokenized_lyrics = [self.START] + re.sub('\n',' %s '%self.EOL, samp['lyrics']).split() + [self.END]
            sample['inp_words'] = tokenized_lyrics[:-1][:self.max_seq_len]
            sample['out_words'] = tokenized_lyrics[1:self.max_seq_len+1]
            sample['inp_ids'] = self.word_tensor(sample['inp_words'])
            sample['out_ids'] = self.word_tensor(sample['out_words'])
        
        if self.use_artist:
            sample['artist'] = samp['artist']
            sample['artist_id'] = self.artists.index(sample['artist'])

        if self.use_melody:
            melody = []
            note_lines = [list(itertools.chain.from_iterable(line)) for line in samp['melody']]
            for note_line in note_lines:
                melody_line = []
                for note in note_line:
                    duration = int(math.ceil(note[1]/0.2))
                    melody_line.extend([note[0]]*duration)
                #if self.use_semantics:
                #    melody_line.extend([0] * self.max_mel_len)
                #    melody.append(melody_line[:self.max_mel_len])
                #else:
                #    melody.extend(melody_line)
                melody_line.extend([0] * self.max_mel_len)
                melody.append(melody_line[:self.max_mel_len])
            #if not self.use_semantics:
            #    melody.extend([0] * (self.max_mel_len))
            #    melody = melody[:self.max_mel_len]
            sample['melody'] = melody

        sample['song'] = samp['song']
    
        return sample
        
    # Turn list of words into list of longs
    def word_tensor(self,words):
        tensor = torch.zeros(len(words)).long()
        for w in range(len(words)):
            tensor[w] = self.word2id(words[w])
        return Variable(tensor)

    def word2id(self, word):
        try:
            idx = self.vocab.index(word)
        except Exception as e:
            idx = self.vocab.index(self.UNK)
        return idx
    
    def id2word(self, idx):
        return self.vocab[idx]


def padding_fn(data):
    # gets samples (dicts) from Data
    
    def merge(seqs):
        lengths = [len(s) for s in seqs]
        max_seq_len = np.max(lengths)
        
        padded_seqs = torch.ones(len(seqs), max_seq_len).long()*PAD_ID
        for i,s in enumerate(seqs):
            end = lengths[i]
            padded_seqs[i, :end] = s[:end]
                
        return padded_seqs, lengths
    
    data.sort(key=lambda x:len(x['inp_ids']),reverse=True)
    
    inp_seqs,inp_lens = merge([x['inp_ids'] for x in data])
    out_seqs,out_lens = merge([x['out_ids'] for x in data])
    
    if data[0]['artist_id'] != []:
        inp_artists = torch.from_numpy(np.array([x['artist_id'] for x in data]))
    else:
        inp_artists = None

    if 'melody' in data[0].keys():
        inp_melody = torch.from_numpy(np.array([x['melody'] for x in data], dtype='float32'))
    else:
        inp_melody=None
        
    return inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,inp_melody,data


def line_padding_fn(data):
    # gets samples (dicts) from Data

    def merge(seqs):
        lengths = [len(s) for s in seqs]
        max_len = np.max(lengths)
        
        padded_seqs = torch.ones(len(seqs), max_len).long()*PAD_ID
        for i,s in enumerate(seqs):
            end = lengths[i]
            padded_seqs[i, :end] = s[:end]
                
        return padded_seqs, lengths
    
    # flatten into a batch of lines (no distinction between song samples)
    # inp_ids = torch.stack([x['inp_ids'] for x in data])
    # out_ids = torch.stack([x['out_ids'] for x in data])
    # inp_ids = inp_ids.view(-1, inp_ids.size()[-1])
    # out_ids = out_ids.view(-1, inp_ids.size()[-1])
    inp_ids = [line for samp in data for line in samp['inp_ids']]
    out_ids = [line for samp in data for line in samp['out_ids']]

    # data.sort(key=lambda x:len(x['inp_ids']),reverse=True)
    
    inp_seqs,inp_lens = merge(inp_ids)
    out_seqs,out_lens = merge(out_ids)
    
    if data[0]['artist_id'] != []:
        inp_artists = torch.from_numpy(np.array([x['artist_id'] for x in data]))
    else:
        inp_artists = None

    if 'melody' in data[0].keys():
        inp_melody = [mel_line for samp in data for mel_line in samp['melody']]
        #inp_melody = [samp['melody'] for samp in data]
        inp_melody = torch.from_numpy(np.array(inp_melody, dtype='float32'))
    else:
        inp_melody=None
        
    return inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,inp_melody,data
