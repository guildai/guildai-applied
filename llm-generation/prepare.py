import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

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

class SongLyrics(Dataset):  
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for row in control_code:
          self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))               
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]
    
dataset = SongLyrics(df['Lyric'], truncate=True, gpt2_type="gpt2")
torch.save(
    dataset,
    "lyrics-dataset.pt",
)