import time

import gradio as gr


def trim_words(words, lens, progress=gr.Progress()):
    trimmed_words = []
    for w, l in progress.tqdm(zip(words, lens)):
        time.sleep(1)
        trimmed_words.append(w[:int(l)])
    return [trimmed_words]


demo = gr.Interface(
    trim_words,
    ["text", "number"],
    ["text"],
    batch=True,
    max_batch_size=16,
)
demo.queue()
demo.launch()
