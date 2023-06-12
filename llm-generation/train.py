import os
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
dataset = torch.load("lyrics-dataset.pt")


batch_size=16
epochs=5
lr=2e-5
max_seq_len=400
warmup_steps=200
gpt2_type="gpt2"
output_dir="."
output_prefix="wreckgar"
test_mode=False
save_model_on_epoch=False

acc_steps = 100
device=torch.device("cuda")
model = model.cuda()
model.train()

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
)

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
loss=0
accumulating_batch_count = 0
input_tensor = None

for epoch in range(epochs):
    for idx, entry in tqdm(enumerate(train_dataloader)):
        input_tensor, carry_on, _ = pack_tensor(entry, input_tensor, 768)

        if carry_on and idx != len(train_dataloader) - 1:
            continue

        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs[0]
        loss.backward()

        if (accumulating_batch_count % batch_size) == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        accumulating_batch_count += 1
        input_tensor = None
    if save_model_on_epoch:
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
        )
    print(f"step: {epoch}")
    print(f"loss: {loss}")

torch.save(
    model.state_dict(),
    os.path.join(output_dir, f"{output_prefix}-final.pt"),
)