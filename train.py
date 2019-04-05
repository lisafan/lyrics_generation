#!/usr/bin/env python
# coding: utf-8

# # Pre-processing data
# Load lyrics with artist info


import sys,os
import string, re
import unidecode
import random, math, time
import pickle
import argparse
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

from data import LyricsDataset, padding_fn, semantic_padding_fn
from model import LyricsRNN
from semantic_model import SemanticLyricsRNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_hyperparameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file")                             # path to python pickle file
    parser.add_argument("--vocab_file")
    parser.add_argument("--checkpoint_files", required=True)        # path and prefix for checkpoints
    parser.add_argument("--load_model")                             # path to saved checkpoint

    parser.add_argument("--vocab_size", default=10000, type=int)    
    parser.add_argument("--chunk_size", default=0, type=int)        # # of lines to use in one input (0 uses the entire song)
    parser.add_argument("--max_seq_len", default=50, type=int)      # max # of words for input and output seq
    parser.add_argument("--max_mel_len", default=40, type=int)      # max # of notes for melody sequences
    parser.add_argument("--max_line_len", default=5, type=int)      # max # of lines for input and output seq when using semantic RNN
    parser.add_argument("--use_artist", default="True", type=str2bool)   # use artist info in input
    parser.add_argument("--use_semantics", default="True", type=str2bool)   # use semantic RNN
    parser.add_argument("--use_melody", default="True", type=str2bool)   # use melody info    
    parser.add_argument("--use_noise", default="False", type=str2bool)   # use noise instead of semantic representation (for baseline model)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_layers_S", default=1, type=int)
    parser.add_argument("--hidden_size_S", default=256, type=int)
    parser.add_argument("--n_layers_L", default=1, type=int)
    parser.add_argument("--hidden_size_L", default=256, type=int)
    parser.add_argument("--word_embedding_size", default=128, type=int)
    parser.add_argument("--artist_embedding_size", default=32, type=int)
    parser.add_argument("--embed_artist", default="False", type=str2bool) # whether to embed artist (T) or use one-hot vector (F)

    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--print_every", default=1000, type=int)
    parser.add_argument("--plot_every", default=1000, type=int)

    args = parser.parse_args()
    return args


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def main():
    # Get hyperparameters from commandline
    params = get_hyperparameters()
    checkpoint_dir = os.path.dirname(params.checkpoint_files)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    log_file = open(params.checkpoint_files+"_output_log.txt",'w')

    def log_str(s):
        print(s)
        log_file.write(s+'\n')
        log_file.flush()

    # echo hyperparameters and run command so it could get logged
    log_str(' '.join(sys.argv)+'\n')
    for k,v in vars(params).items():
        log_str(k+": "+str(v))

    # Load previous hyperparameters if checkpoint was given
    if params.load_model != None:
        checkpoint = torch.load(params.load_model)
        params = checkpoint['hyperparameters']

    # --------------
    # Get data (train & val)
    Data = LyricsDataset(params.input_file, vocab_file=params.vocab_file, vocab_size=params.vocab_size,
                         chunk_size=params.chunk_size, max_line_len=params.max_line_len, max_seq_len=params.max_seq_len,
                         use_semantics=params.use_semantics, use_artist=params.use_artist)
    # print(Data[np.random.randint(len(Data))], len(Data))
    # exit()
    log_str("\n%d batches per epoch\n"%(len(Data)/params.batch_size))

    if params.use_semantics:
        pad_fn = semantic_padding_fn
    else:
        pad_fn = padding_fn
    dataloader = DataLoader(Data, batch_size=params.batch_size, shuffle=True, num_workers=0, collate_fn=pad_fn, drop_last=True)
    # for i,batch in enumerate(dataloader):
    #     print(batch[1])
    #     exit()
    #     break
        
    ValData = LyricsDataset(re.sub('train','val',params.input_file), vocab_file=params.vocab_file, 
                            chunk_size=params.chunk_size,max_line_len=params.max_line_len,
                            max_seq_len=params.max_seq_len, max_mel_len=params.max_mel_len,
                            use_semantics=params.use_semantics, use_artist=params.use_artist,
                            use_melody=params.use_melody)
    val_dataloader = DataLoader(ValData,  batch_size=params.batch_size, num_workers=0, collate_fn=pad_fn, drop_last=True)

    # --------------
    # Create model and optimizer
    if params.use_semantics:
        model = SemanticLyricsRNN(Data.vocab_len, Data.vocab_len, Data.PAD_ID, batch_size=params.batch_size, 
                            n_layers_S=params.n_layers_S, hidden_size_S=params.hidden_size_S, n_layers_L=params.n_layers_L, 
                            hidden_size_L=params.hidden_size_L, word_embedding_size=params.word_embedding_size,
                            use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists, 
                            artist_embedding_size=params.artist_embedding_size, use_noise=params.use_noise
                          ).to(device)
    else:
        model = LyricsRNN(Data.vocab_len, Data.vocab_len, Data.PAD_ID, batch_size=params.batch_size, 
                            n_layers=params.n_layers_L, hidden_size=params.hidden_size_L, word_embedding_size=params.word_embedding_size,
                            use_artist=params.use_artist, embed_artist=params.embed_artist, num_artists=Data.num_artists, 
                            artist_embedding_size=params.artist_embedding_size
                          ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    
    # Load checkpoint
    if params.load_model != None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # --------------
    # Helper functions
    def generate(prime_str=[Data.START], artist=None, predict_line_len=5, predict_seq_len=20, temperature=0.8):
        if type(artist)==str:
            artist = Data.artists.index(artist)

        if params.use_semantics:
            if type(prime_str[0])==list:
                inp = [[Data.word2id(w) for w in prime_line] for prime_line in prime_str]
            else:
                inp = [[Data.word2id(w) for w in prime_str]]*predict_line_len
            predicted = model.evaluate(inp, artist, predict_line_len, predict_seq_len, temperature)

            predicted_words = []
            for line in predicted:
                pw = [Data.id2word(w) for w in line]
                if Data.EOL in pw:
                    pw = pw[:pw.index(Data.EOL)+1]
                predicted_words += pw + ['\n']

        else:
            inp = [Data.word2id(w) for w in prime_str]
            predict_len = predict_line_len*predict_seq_len
            predicted = model.evaluate(inp, artist, predict_len, temperature)

            predicted_words = [Data.id2word(w) for w in predicted]
            if Data.END in predicted_words:
                predicted_words = predicted_words[:predicted_words.index(Data.END)+1]

        return ' '.join(predicted_words)

    def check_early_stopping(prev_val_loss):
        log_str("Starting validation check...")
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i,batch in enumerate(val_dataloader):
                inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,data = batch
                
                if Data.use_artist:
                    inp, target = [inp_seqs.to(device),inp_artists.to(device)], out_seqs.to(device)
                else:
                    inp, target = inp_seqs.to(device), out_seqs.to(device)
                
                predictions = model(inp, inp_lens)
                loss = model.loss(predictions, target)
                val_loss += loss
        avg_val_loss = val_loss / len(val_dataloader)
        log_str('Validation loss: %.4f'%avg_val_loss)
        model.train()
        return avg_val_loss

    # --------------
    # Begin training
    start = time.time()
    all_losses = []
    loss_avg = 0
    prev_val_loss = float('inf')
    epsilon = 0 # gap for early stopping
    start_epoch = 1

    # Load checkpoint
    if params.load_model != None:
        start_epoch = checkpoint['epoch']
        all_losses = checkpoint['loss']

    for epoch in range(start_epoch, params.n_epochs + 1):
        for i, batch in enumerate(dataloader):
            inp_seqs,inp_lens,out_seqs,out_lens,inp_artists,data = batch
            
            # print(' '.join([Data.id2word(x) for x in inp_seqs[0]]))
            if params.use_artist:
                inp, target = [inp_seqs.to(device),inp_artists.to(device)], out_seqs.to(device)
            else:
                inp, target = inp_seqs.to(device), out_seqs.to(device)

            model.zero_grad()
            predictions = model(inp, inp_lens)
            loss = model.loss(predictions, target)
            loss.backward()
            optimizer.step()
            
            loss_avg += loss.detach()

            if i % params.print_every == 0:
                loss_avg /= params.print_every
                all_losses.append(loss_avg / params.print_every)
                log_str('[%s (epoch %d: %d%%) Loss: %.4f]' % (time_since(start), epoch, i / (len(Data.lyrics)/params.batch_size) * 100, loss_avg))
                loss_avg = 0

                if params.use_artist:
                    for a in sorted(Data.artists):
                        log_str('Artist %s: %s\n'%(a, generate(artist=a)))
                else:
                    log_str(generate()+'\n')

                cp_output = predictions
                probs = np.exp(cp_output.cpu().detach().numpy())
                # print('target', ' '.join([Data.id2word(target[0][i]) for i in range(len(target[0]))]))
                # print('max_probs',np.amax(probs[0][0]))
                # print('target_id',target[0])
                # print('predicted_id',[np.argmax(probs[0][i]) for i in range(len(target[0]))])
                # print('prob_target',[probs[0][i][target[0][i]] for i in range(len(probs[0]))])
                # print('eol_prob', [probs[0][i][3] for i in range(len(probs[0]))])
                # print('pred', ' '.join([Data.id2word(np.argmax(probs[0][i])) for i in range(len(target[0]))]))
                # print(np.sum(probs[0][0]))

        # all_losses.append(loss_avg / len(dataloader))
        # log_str("Average epoch loss: %.4f"%(loss_avg/len(dataloader)))
        # loss_avg = 0
    
        check_early_stopping(prev_val_loss)
        # cur_val_loss = check_early_stopping(prev_val_loss)
        # if cur_val_loss > prev_val_loss + epsilon:
        #     print("Early stopping")
        #     break
        # prev_val_loss = cur_val_loss
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': all_losses,
            'hyperparameters': params
        }, '%s-e%05d.pt'%(params.checkpoint_files, epoch))


main()
