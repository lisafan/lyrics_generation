import csv, json, pickle
from collections import Counter, defaultdict
from string import punctuation
import numpy as np

def csv2pkl():
    with open('lyrics.csv') as csvfile:
        data = []
        spamreader = csv.reader(csvfile, quotechar='"')
        for i,row in enumerate(spamreader):
            if i==0: 
                keys = row
                continue
            if i%10001==0:
                print i
                break
            entry = {}
            for k,v in zip(keys,row):
                entry[k] = v
            entry['num_lines'] = len(entry['lyrics'].splitlines())
            data += [entry]
    outfile = open("lyrics_subset.pkl",'w')
    pickle.dump(data,outfile)

# csv2pkl()
data = pickle.load(open("lyrics_top_artists.pkl"))

# artists = defaultdict(list)
# for e in data:
#     artists[e['artist']] += [e]
# for a in artists.keys():
#     print a
#     for s in sorted(s['song'] for s in artists[a]):
#         print s
#     print '\n\n'

# print 'data loaded'
# top_artists = open('top_artists.txt').read().splitlines()
# subset = []
# for e in data:
#     if e['artist'] in top_artists and e['num_lines']>5:
#         subset += [e]

# # print sorted(subset,key=lambda x:x['artist'])
# print len(subset)

outfile = open('lyrics_top_artists.csv','w')
keys = data[0].keys()
keys.remove('lyrics')

for k in keys:
    outfile.write(k+', ')
outfile.write('lyrics\n')
for e in sorted(data,key=lambda x:x['artist']):
    for k in keys:
        outfile.write(str(e[k])+', ')
    outfile.write('"'+e['lyrics']+'"\n')

# pickle.dump(subset,outfile)
# artists = Counter([e['artist'] for e in data])
# top_artists = [x for x in artists.most_common() if x[1]>=100] #300]
# print top_artists, len(top_artists)
# print np.sum([x[1] for x in top_artists])

# for a in top_artists:
#     beyonce_songs = []
#     for e in data:
#         if e['artist']==a[0]:
#             if not any([e['song'].endswith(x) for x in ['remix','version','edition']]):
#                 beyonce_songs += [e['song']]
#     print a
#     for s in sorted(beyonce_songs):
#         print s
#     print '\n\n'
    # exit()
# print artists



exit()

print "read data"
print len(data)

vocab = []
len_lines = []
len_song = []
for i,e in enumerate(data):
    if i%50000==0:
        print i
    vocab += [w.lower().rstrip(punctuation) for w in e['lyrics'].split()]
    # len_lines += [len(l.split()) for l in e['lyrics'].splitlines()]
    # len_song += [e['num_lines']]
vocab = Counter(vocab)
print vocab.most_common(100)

# print "avg words/line: ",np.mean(len_lines)
# print "avg lines/song",np.mean(len_song)

with open('vocab_top_artists.count','w') as f:
    for i,(a,n) in enumerate(vocab.most_common()):
        if i%10000==0:
            print i
        if i==100000:
            break
        if n < 5:
            break
        f.write('%s\t%s\n'%(a,n))

# print artists.most_common()
# print len(artists)
