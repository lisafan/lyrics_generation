# Coherent Lyrics Generation Conditioned on Melody and Artist
Final project for CS7180, spring 2019

This model uses Python 3 and PyTorch version 0.4.0.


To begin training, run:
```
python train.py --input_file=<PATH/INPUT.pkl> <other arguments>
```

The program expects a python pickle file, which can be generated using `lyrics/clean_lyrics.py`.

### Code files
**clean_lyrics.py:** This program takes in a CSV file that requires items to have fields for song title, artist, and lyrics. 
The program also filters songs that have fewer than 5 lyric lines and songs with certain keywords in the title
(ex. songs that end in "version", like "hold-you-acoustic-version", are filtered out).

**train.py:** Script that oversees training. See the beginning of the file for a list of acceptable input arguments and what they mean.

**data.py:** Contains the LyricsDataset class, used to manage the data. Creates vocabulary if none is provided, splits data into chunks of consecutive lines, processes the melody data for input to the model. Also contains the padding functions used by the dataloader.

**model.py:** Contains the LyricsRNN class, the non-hierarchical version of our model. Used to train lyrics-only, melody-only, and artist-embed

**semantic_model.py:** Contains the SemanticLyricsRNN class, the hierarchical model. Used to train with-semantics, no-semantics, and melody-only.

**eval.py:** Script to run automatic evaluations. Calculates perplexity and generates 50 samples. If artist embeddings were used, calculates cosine similarity between select artists, perplexity when the artist and target lyrics are mismatched, and generates png graphs of PCA and TSNE on the artist embeddings. Expects the paths to a saved model, a test data file, and the evaluation output file name.

### Directories
**lyrics:** Contains input files, vocab files, etc.

**checkpoints:** Contains trained models and training logs.

**eval:** Contains results of evaluations, generated from running `eval.py`.

### Trained Models
**lyrics-only:** The simplest baseline. The model is not conditioned on the artist and the second LSTM, is absent. Instead, the output from the first RNN is fed directly into the linear layer.

**artist-onehot:** Input to the LSTM is a concatenation of the previous word's embedding and a one-hot encoding of the song's artist.

**artist-embed:** Same as artist-onehot, but uses learned artist embeddings instead of a one-hot encoding.

**melody-only:** An LSTM that takes in a melody context vector as input and generates aligned lyrics.

**no-semantics:** A baseline for the hierarchical model, where the semantic generator is not used and the lyrics generator is given random noise as input.

**with-semantics:** A hierarchical model with the melody context vector omitted from the input.

**semantic-melody:** The full hierarchical model where the semantics generator's output is fed into the lyrics generator, along with the melody context vector.
