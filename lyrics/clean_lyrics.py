import os, sys, re
import csv, json, pickle
import numpy as np
from collections import Counter, defaultdict
from string import punctuation
from nltk.tokenize import word_tokenize

top_artists = open('top_artists.txt').read().splitlines()
stopwords = ['version', 'remix', 'edition']

# conditions:
c1 = lambda x: x in top_artists                                     # song by a top artist
c2 = lambda x,y: len(re.findall('\n',x))>y                          # lyrics have at least y lines
c3 = lambda x: not any([x.endswith(stop) for stop in stopwords])    # title doesn't end in any keywords

def csv2pkl():
    with open('lyrics.csv') as csvfile:
        data = []
        spamreader = csv.reader(csvfile, quotechar='"')
        for i,row in enumerate(spamreader):
            if i==0: 
                # ['index', 'song', 'year', 'artist', 'genre', 'lyrics']
                keys = row
                continue
            if i%10000==0:
                print (i)

            if c1(row[3]) and c2(row[5],5) and c3(row[1]):
                entry = {}
                for k,v in zip(keys,row):
                    if k=='lyrics':
                        entry[k] = clean_lyrics(v)
                    else:
                        entry[k] = v
                entry['num_lines'] = len(entry['lyrics'].splitlines())
                data += [entry]
    print(len(data))
    outfile = open("lyrics_top_artists.pkl",'wb')
    pickle.dump(data,outfile)

def clean_lyrics(lyr):
    lines = lyr.split('\n')
    tokens = [word_tokenize(l.lower()) for l in lines]
    tokens = '\n'.join([' '.join(l) for l in tokens])
    return tokens

# csv2pkl()

def split_sets(infile, outfile):
    # infile = 'lyrics_top_artists.pkl'
    data = pickle.load(open(infile,'rb'))
    print (len(data))
    train_cutoff = int(len(data)*.8)
    val_cutoff = int(len(data)*.9)
    print(train_cutoff,val_cutoff)
    # outfile = 'artist'
    with open('%s_train.pkl'%outfile,'wb') as f:
        pickle.dump(data[:train_cutoff],f)
    with open('%s_val.pkl'%outfile,'wb') as f:
        pickle.dump(data[train_cutoff:val_cutoff],f)
    with open('%s_test.pkl'%outfile,'wb') as f:
        pickle.dump(data[val_cutoff:],f)

    d1 = pickle.load(open('%s_train.pkl'%outfile,'rb'))
    d2 = pickle.load(open('%s_val.pkl'%outfile,'rb'))
    d3 = pickle.load(open('%s_test.pkl'%outfile,'rb'))
    print(len(d1),len(d2),len(d3))
# split_sets()


def one_artist():
    data = pickle.load(open('lyrics_top_artists.pkl','rb'))
    dolly = []
    for d in data:
        if d['artist'] in ['dolly-parton', 'elton-john', 'b-b-king', 'chris-brown', 'eminem']:
            dolly += [d]
    pickle.dump(dolly,open('top_5.pkl','wb'))

one_artist()
split_sets('top_5.pkl','top-5')