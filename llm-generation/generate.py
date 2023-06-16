import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


entry_count=10
entry_length=30
top_p=0.8
temperature=1.0
output_dir="."
output_prefix="wreckgar"
test_set_path="test_set.pkl"

def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
                generated_list.append(output_text)
                
    return generated_list

#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data, model, tokenizer):
    generated_lyrics = []
    for i in range(len(test_data)):
        x = generate(model.to('cpu'), tokenizer, test_data['Lyric'][i], entry_count=1)
        generated_lyrics.append(x)
    return generated_lyrics

#Run the functions to generate the lyrics
test_set = pd.read_pickle(test_set_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
state_dict = torch.load(
    os.path.join(output_dir, f"{output_prefix}-final.pt")
)
model = GPT2LMHeadModel.from_pretrained(
    'gpt2',
    state_dict=state_dict,
)

generated_lyrics = text_generation(
   test_set,
   model,
   tokenizer,
)

generations=[]
lyrics = test_set['Lyric']
for i in range(len(generated_lyrics)):
    generated = generated_lyrics[i][0]
    original = lyrics[i]
    new = generated[len(original):].split("<")[0].split(".")[0].strip()
    generations.append(new)

test_set['Generated_lyrics'] = generations
test_set.to_pickle("generated_set.pkl")