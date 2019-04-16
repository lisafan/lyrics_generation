import os, sys, re
import csv, json, pickle
import random
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
    random.shuffle(data)
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
# split_sets("input_files/top_5.pkl","input_files/new_top5")


def select_artists(artists, inp_pkl, out_pkl):
    data = pickle.load(open(inp_pkl,'rb'), encoding='latin1')
    subset = []
    for d in data:
        if d['artist'] in artists:
            subset += [d]
    pickle.dump(subset,open(out_pkl,'wb'))

# one_artist()
# # split_sets('top_5.pkl','top-5')
# artists = [re.sub(' ','-', a) for a in open('artist_list.txt').read().splitlines()]
# select_artists(artists, 'large_files/lyrics.pkl', 'input_files/filtered_kaggle.pkl')
# split_sets('input_files/filtered_kaggle.pkl', 'input_files/filtered_kaggle')

def create_vocab(lyrics,file_name):
    num_songs = len(lyrics)
    print('creating vocabulary for %d songs'%num_songs)
    
    vocab = []
    for i,e in enumerate(lyrics):
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

split_sets('input_files/filtered_kaggle.pkl', 'input_files/filtered_kaggle')
kaggle = pickle.load(open('input_files/filtered_kaggle_train.pkl','rb'))
create_vocab(kaggle, 'input_files/filtered_kaggle_train.vocab')

# dali = pickle.load(open('input_files/filtered_dali_train.pkl','rb'))
# kaggle = pickle.load(open('input_files/filtered_kaggle.pkl','rb'))
# print(len(kaggle))
# kaggle = pickle.load(open('input_files/kaggle_testing.pkl','rb'))
# print(kaggle)
# outfile = open('input_files/filtered_kaggle2.pkl','wb')
# songs = []
# for k in kaggle:
#     k['lyrics'] = clean_lyrics(k['lyrics'])
#     songs += [k.copy()]
# pickle.dump(songs,outfile)

# print(kaggle[5]['artist'])
# select_artists(['frank-sinatra'],'input_files/filtered_kaggle.pkl','input_files/sinatra.pkl')
# sinatra = pickle.load(open('input_files/sinatra.pkl','rb'))
# print(len(sinatra))
# split_sets('input_files/sinatra.pkl','input_files/sinatra')
# artist_title = set()
# lyrics = []
# for x in dali+kaggle:
#     at = re.sub(' ','-',x['artist'])+'-'+re.sub(' ','-',x['song'])
#     if at not in artist_title:
#         artist_title.add(at)
        # lyrics += [x]

# create_vocab(lyrics,'input_files/filtered.vocab')


