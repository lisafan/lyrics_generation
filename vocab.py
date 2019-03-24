import os,sys,re
import pickle,json
from collections import Counter

train = pickle.load(open('lyrics/input_files/dolly_train.pkl','rb'))
val = pickle.load(open('lyrics/input_files/dolly_val.pkl','rb'))

train_words=[]
for t in train:
	train_words += t['lyrics'].split()

val_words = []
for v in val:
	val_words += v['lyrics'].split()

train_count = Counter(train_words)
val_count = Counter(val_words)
print(train_count['blurred'])

train_count = [x for x in train_count.most_common() if x[1]>5]
# val_count = [x in val_count.most_common() if x[1]>5]
val_unk = set(val_count.keys())-set([x[0] for x in train_count])
print(len(val_unk))
print(len(val_count))
print(len(val_words))
print(sum([val_count[x] for x in list(val_unk)]))
print(list(val_unk)[0])
print(train_words.index('blurred'))