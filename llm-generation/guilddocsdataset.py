import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class GuildDocs(Dataset):  
    def __init__(
        self,
        docs_frame,
        truncate=False,
        gpt2_type="gpt2",
        max_length=1024,
    ):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.docs = []

        for doc in docs_frame:
            doc_text = f"<|startoftext|>{doc[:max_length]}<|endoftext|>"
            self.docs.append(torch.tensor(self.tokenizer.encode(doc_text)))
            
        if truncate:
            self.docs = self.docs[:20000]
        self.docs_count = len(self.docs)
        
    def __len__(self):
        return self.docs_count

    def __getitem__(self, item):
        return self.docs[item]