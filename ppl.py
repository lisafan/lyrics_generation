import sys,os
from model import LyricsRNN
from data import LyricsDataset, padding_fn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(sys.argv[1]) #, map_location=device)
epoch = checkpoint['epoch']
all_losses = checkpoint['losses']
params = checkpoint['hyperparameters']

Data = LyricsDataset(sys.argv[2], vocab_file=sys.argv[3],
                        chunk_size=params.chunk_size, use_artist=params.use_artist)
val_dataloader = DataLoader(Data, batch_size=params.batch_size, num_workers=1, collate_fn=padding_fn, drop_last=True)

# Create model and optimizer
model = LyricsRNN(Data.vocab_len, Data.vocab_len, Data.PAD_ID, batch_size=params.batch_size, n_layers=params.n_layers,
                  hidden_size=params.hidden_size, word_embedding_size=params.word_embedding_size,
                  use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists,
                  artist_embedding_size=params.artist_embedding_size
                  )
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(len(val_dataloader))


def generate(prime_str=[Data.START], artist=None, predict_len=100, temperature=0.8):
    inp = [Data.word2id(w) for w in prime_str]

    if type(artist)==str:
        artist = Data.artists.index(artist)

    predicted = model.evaluate(inp, artist, predict_len, temperature)

    predicted_words = [Data.id2word(w) for w in predicted]
    if Data.END in predicted_words:
        predicted_words = predicted_words[:predicted_words.index(Data.END)+1]

    return ' '.join(predicted_words)


print(generate())

def evaluate(model = None, val_dataloader = None):
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            inp_seqs, inp_lens, out_seqs, out_lens, inp_artists, data = batch
            if params.use_artist:
                inp, target = [inp_seqs.to(device), inp_artists.to(device)], out_seqs.to(device)
            else:
                inp, target = inp_seqs.to(device), out_seqs.to(device)
            predictions = model(inp, inp_lens)
            loss = model.loss(predictions, target)
            #loss = loss_func(predictions.view(-1, predictions.size(2)), target.view(-1).long())
            val_loss += loss.item()
            #loss = loss_func(predictions, target)
            #loss = model.loss(predictions, target)
            if i % 100 == 0:
                print({},i/len(val_dataloader))
        val_loss /= len(val_dataloader)
        print('ppl: {:5.2f},'.format(np.exp(val_loss)))
evaluate(model, val_dataloader)