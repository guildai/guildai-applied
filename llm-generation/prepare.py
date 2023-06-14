import pandas as pd
import torch
# from torch.utils.data import Dataset
# from transformers import GPT2Tokenizer
from songlyricsdataset import SongLyrics

### Prepare data
lyrics = pd.read_csv('lyrics-data.csv')
lyrics = lyrics[lyrics['language']=='en']

#Only keep popular artists, with genre Rock/Pop and popularity high enough
artists = pd.read_csv('artists-data.csv')
artists = artists[(artists['Genres'].isin(['Rock'])) & (artists['Popularity']>5)]
df = lyrics.merge(
    artists[['Artist', 'Genres', 'Link']],
    left_on='ALink',
    right_on='Link',
    how='inner',
)
df = df.drop(columns=['ALink','SLink','language','Link'])

#Drop the songs with lyrics too long (after more than 1024 tokens, does not work)
df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350)]

#Create a very small test set to compare generated text with the reality
test_set = df.sample(n = 200)
df = df.loc[~df.index.isin(test_set.index)]

#Reset the indexes
test_set = test_set.reset_index()
df = df.reset_index()

#For the test set only, keep last 20 words in a new column, then remove them from original column
end_len = 20
test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-end_len:].apply(' '.join)
test_set['Lyric'] = test_set['Lyric'].str.split().str[:-end_len].apply(' '.join)
test_set.to_pickle("test_set.pkl")
    
dataset = SongLyrics(df['Lyric'], truncate=True, gpt2_type="gpt2")
torch.save(
    dataset,
    "lyrics-dataset.pt",
)