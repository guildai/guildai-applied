import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


entry_count=1
entry_length=100 #maximum number of words
top_p=0.8
temperature=1.0
prompt="test prompt"
ending = "<|endoftext|>"


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
    ending="<|endoftext|>",
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for _ in range(entry_count):

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

                if next_token in tokenizer.encode(ending):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}{ending}" 
                generated_list.append(output_text)
                
    return generated_list


def gradio_fn(
    prompt=prompt,
    entry_length=entry_length,
    temperature=temperature,
    top_p=top_p,
):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    generated = generate(
        model.to('cpu'),
        tokenizer,
        prompt,
        entry_count=entry_count,
        entry_length=entry_length,
        top_p=top_p,
        temperature=temperature,
        ending=ending,
    )
    return generated[0].replace(ending, "")

import app as gr

app = gr.Interface(
    fn=gradio_fn,
    inputs=[
        gr.TextArea("test prompt"),
        gr.Slider(1,1024,100,step=1),
        gr.Slider(0.01,1,1.0,step=0.01),
        gr.Slider(0.01,1,0.8,step=0.01),
    ],
    outputs="text",
)

app.launch()