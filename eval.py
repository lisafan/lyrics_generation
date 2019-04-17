import sys,os,re
import random
import argparse
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import linalg
import torch
from torch.utils.data import Dataset, DataLoader
from model import LyricsRNN
from data import LyricsDataset, padding_fn, line_padding_fn

matplotlib.use('Agg')
torch.cuda.set_device(1)
print(torch.cuda.get_device_name(1))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_hyperparameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint")
    parser.add_argument("--testfile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    return args


def generate(model, Data, params, prime_str=None, artist=None, melody=None, pad_fn=None, predict_line_len=5, predict_seq_len=20, temperature=0.8):
    if prime_str==None:
        prime_str = [Data.START]

    if type(artist)==str:
        artist = Data.artists.index(artist)

    if params.use_melody:
        if melody==None:
            _,_,_,_,_,sample_melody,sample = pad_fn([Data[random.randint(0,len(Data))]])
            melody = sample_melody.to(device)
            print('Melody source: %s by %s\n'%(sample[0]['song'], sample[0]['artist']))
    else:
        sample_melody = None

    if params.use_semantics:
        if type(prime_str[0])==list:
            inp = [[Data.word2id(w) for w in prime_line] for prime_line in prime_str]
        else:
            inp = [[Data.word2id(w) for w in prime_str]]*predict_line_len
        predicted = model.evaluate(inp, artist, melody, predict_line_len, predict_seq_len, temperature).detach()

        predicted_words = []
        for line in predicted:
            pw = [Data.id2word(w) for w in line]
            if Data.EOL in pw:
                pw = pw[:pw.index(Data.EOL)+1]
            predicted_words += pw + ['\n']

    else:
        inp = [Data.word2id(w) for w in prime_str]
        predict_len = predict_line_len*predict_seq_len
        predicted = model.evaluate(inp, artist, melody, predict_len, temperature)

        predicted_words = [Data.id2word(w) for w in predicted]
        if Data.END in predicted_words:
            predicted_words = predicted_words[:predicted_words.index(Data.END)+1]

    return ' '.join(predicted_words)


def perplexity(model, val_dataloader, params):
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,inp_melody,data = batch
            if params.use_artist:
                if params.use_melody:
                    inp, target = [inp_seqs.to(device),inp_melody.to(device),inp_artists.to(device)], out_seqs.to(device)
                else:
                    inp, target = [inp_seqs.to(device),inp_artists.to(device)], out_seqs.to(device)
            else:
                if params.use_melody:
                    inp, target = [inp_seqs.to(device),inp_melody.to(device)], out_seqs.to(device)
                else:
                    inp, target = inp_seqs.to(device), out_seqs.to(device)
            predictions = model(inp, inp_lens)
            loss = model.loss(predictions, target)
            val_loss += loss.item()
            if i % 100 == 0:
                print('%.2f%% done'%(i/len(val_dataloader)*100))
        val_loss /= len(val_dataloader)
        ppl = np.exp(val_loss)
        return ppl

def artist_perplexity(model, val_dataloader, params, num_artists):
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,inp_melody,data = batch

            for j in range(len(inp_artists)):
                rand = inp_artists[j]
                while rand==inp_artists[j]:
                    rand = random.randint(0,num_artists-1)
                inp_artists[j] = rand

            if params.use_artist:
                if params.use_melody:
                    inp, target = [inp_seqs.to(device),inp_melody.to(device),inp_artists.to(device)], out_seqs.to(device)
                else:
                    inp, target = [inp_seqs.to(device),inp_artists.to(device)], out_seqs.to(device)
            else:
                if params.use_melody:
                    inp, target = [inp_seqs.to(device),inp_melody.to(device)], out_seqs.to(device)
                else:
                    inp, target = inp_seqs.to(device), out_seqs.to(device)

            predictions = model(inp, inp_lens)
            loss = model.loss(predictions, target)
            val_loss += loss.item()
            if i % 100 == 0:
                print('%.2f%% done'%(i/len(val_dataloader)*100))
        val_loss /= len(val_dataloader)
        ppl = np.exp(val_loss)
        return ppl

def load_artist_vectors(filename):
    checkpoint = torch.load(filename, map_location=device)
    epoch = checkpoint['epoch']
    all_losses = checkpoint['losses']
    params = checkpoint['hyperparameters']
    Data = LyricsDataset(params.input_file, vocab_file=params.vocab_file, vocab_size=params.vocab_size,
                     chunk_size=params.chunk_size, max_len=params.max_seq_len,
                     use_artist=params.use_artist)
    ValData = LyricsDataset(re.sub('train', 'val', params.input_file), vocab_file=params.vocab_file,
                        chunk_size=params.chunk_size, use_artist=params.use_artist)
    val_dataloader = DataLoader(ValData, batch_size=params.batch_size, num_workers=1, collate_fn=padding_fn, drop_last=True)
    model = LyricsRNN(ValData.vocab_len, ValData.vocab_len, ValData.PAD_ID, batch_size=params.batch_size, n_layers=params.n_layers,
                  hidden_size=params.hidden_size, word_embedding_size=params.word_embedding_size,
                  use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists,
                  artist_embedding_size=params.artist_embedding_size
                  )
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for var_name in model.state_dict():
        print(var_name)
    print(model.artist_encoder.weight)
    artist_labels = Data.artists
    arr = model.artist_encoder.weight.detach().numpy()
    embed_artists = {}
    for index, artist in enumerate(Data.artists):
        embed_artists[artist] = arr[index,:]
    print(embed_artists)
    return artist_labels, arr, embed_artists

def plot_embeddings(artist_labels, arr, filename, mode):
    if mode=='tsne':
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        lim = 10

    elif mode=='pca':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(arr)
        lim=1

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.clf()
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(artist_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(2, 2), textcoords='offset points', fontsize=6)
    plt.xlim(x_coords.min()-lim, x_coords.max()+lim)
    plt.ylim(y_coords.min()-lim, y_coords.max()+lim)
    plt.savefig('%s_%s.png'%(filename,mode))
    # plt.figure()
    # plt.show()


def cos(vec1,vec2):
    return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

# cosine similarity of input artist against all others
def similarity(a, embed_artists):
    sim = []
    for b in embed_artists.keys():
        if b==a:
            continue
        res = cos(embed_artists[a], embed_artists[b])
        sim += [[b, res]]
    # for a, b in pairs:
    #     res = cos(embed_artists[a], embed_artists[b])
    #     sim[(a,b)] = res
        # print(res,a,'and',b)
    return(sim)


def main():
    args = get_hyperparameters()
    checkpoint = torch.load(args.checkpoint)
    epoch = checkpoint['epoch']
    all_losses = checkpoint['losses']
    params = checkpoint['hyperparameters']

    # print(params)

    Data = LyricsDataset(args.testfile, vocab_file=params.vocab_file, chunk_size=params.chunk_size, 
        max_line_len=params.max_line_len, max_seq_len=params.max_seq_len, max_mel_len=params.max_mel_len, 
        use_semantics=params.use_semantics, use_artist=params.use_artist, use_melody=params.use_melody, artists='lyrics/artist_list.txt')

    if params.use_semantics or params.use_melody:
        pad_fn = line_padding_fn
    else:
        pad_fn = padding_fn
    dataloader = DataLoader(Data, batch_size=params.batch_size, shuffle=False, num_workers=0, collate_fn=pad_fn, drop_last=True)


    if params.use_semantics:
        model = SemanticLyricsRNN(Data.vocab_len, Data.vocab_len, Data.PAD_ID, batch_size=params.batch_size, 
                            n_layers_S=params.n_layers_S, hidden_size_S=params.hidden_size_S, n_layers_L=params.n_layers_L, 
                            hidden_size_L=params.hidden_size_L, melody_len=params.max_mel_len,
                            word_embedding_size=params.word_embedding_size,
                            use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists, 
                            artist_embedding_size=params.artist_embedding_size, use_noise=params.use_noise,
                            use_melody=params.use_melody
                          )
    else:
        model = LyricsRNN(Data.vocab_len, Data.vocab_len, Data.PAD_ID, batch_size=params.batch_size, 
                            n_layers=params.n_layers_L, hidden_size=params.hidden_size_L, melody_len=params.max_mel_len,
                            word_embedding_size=params.word_embedding_size,
                            use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists, 
                            artist_embedding_size=params.artist_embedding_size, use_melody=params.use_melody
                          )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    ppl = perplexity(model, dataloader, params)

    if params.use_artist:
        artist_ppl = artist_perplexity(model, dataloader, params, Data.num_artists)

    if 'artist_encoder.weight' in model.state_dict().keys():
        artist_embeddings =  model.artist_encoder.weight.detach()
        emb_dict = {}
        for artist,emb in zip(Data.artists,artist_embeddings):
            emb_dict[artist] = emb

        sims = {}
        for artist in ['britney spears', 'bruce springsteen', 'the gathering', 'elton john']:
            sim = similarity(artist, emb_dict)
            sims[artist] = sorted(sim, key=lambda x: x[1], reverse=True)

    samples = []
    for i in range(50):
        if params.use_artist:
            idx = random.randint(0,Data.num_artists-1)
            ex = generate(model, Data, params, artist=idx, pad_fn=pad_fn)
            ex = Data.artists[idx]+':\n' + re.sub(' <EOL> ','\n', ex)
        else:
            ex = generate(model, Data, params, pad_fn=pad_fn)
            ex = re.sub(' <EOL> ','\n', ex)
        samples += [ex]

    with open(args.outfile, 'w') as f:
        f.write('Evaluations for %s\n'%args.checkpoint)
        f.write('using %s:\n\n'%args.testfile)
        f.write('Perplexity = %.3f\n'%ppl)
        if params.use_artist:
            f.write('Wrong Artist Perplexity = %.3f\n'%artist_ppl)

        if 'artist_encoder.weight' in model.state_dict().keys():
            f.write('\nSelect cosine similarities:\n')
            for k in sims.keys():
                f.write('%s\n'%k)

                f.write('  Top 3 most common:\n')
                for j in sims[k][:3]:
                    f.write('%20s : %.4f\n'%(j[0],j[1]))

                f.write('\n  Top 3 least common:\n')
                for j in sims[k][-3:]:
                    f.write('%20s : %.4f\n'%(j[0],j[1]))
                f.write('\n\n')

        f.write('\n50 generated samples:\n\n')
        for i, s in enumerate(samples):
            f.write('(%d)\n'%(i+1))
            f.write(s+'\n\n')
main()