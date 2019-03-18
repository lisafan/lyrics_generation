# Coherent Lyrics Generation Conditioned on Melody and Artist
Final project for CS7180, spring 2019

This model uses Python 3 and PyTorch version 0.4.0.

To begin training, run:
```
python train.py --input_file=<PATH/INPUT.pkl> <other arguments>
```

The program expects a python pickle file, which can be generated using `lyrics/clean_lyrics.py`.

### clean_lyrics.py
This program takes in a CSV file that requires items to have fields for song title, artist, and lyrics. 
The program also filters songs that have fewer than 5 lyric lines and songs with certain keywords in the title
(ex. songs that end in "version", like "hold-you-acoustic-version", are filtered out).

### Current Models
#### dolly
Trained only on Dolly Parton lyrics (755 songs). Not conditioned on artist.

#### top5_onehot
Conditioned on 5 common artists (Dolly Parton, Elton John, BB King, Chris Brown, and Eminem).
These artists each had at least 550 songs in our dataset.
The model was trained using one-hot encodings for the 5 artists.

#### top5_embed
Conditioned on 5 common artists (Dolly Parton, Elton John, BB King, Chris Brown, and Eminem). 
The model was trained to learn 64 bit embeddings for the artists.
