import transformers

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

def _maybe_add(args, key, val):
    if val is not None:
        args[key] = val

def _required(key, val):
    if not val:
        raise TypeError(f"{key} can't be {val}")

args = dict()

prompt = "This is the default prompt"
_required("prompt", prompt)
inputs = tokenizer.encode(prompt, return_tensors='pt')
_maybe_add(args, "inputs", inputs)

do_sample=None
_maybe_add(args, "do_sample", do_sample)

early_stopping=None
_maybe_add(args, "early_stopping", early_stopping)

max_length=50
_maybe_add(args, "max_length", max_length)

no_repeat_ngram_size=None
_maybe_add(args, "no_repeat_ngram_size", no_repeat_ngram_size)

num_beams=None
_maybe_add(args, "num_beams", num_beams)

num_return_sequences=None
_maybe_add(args, "num_return_sequences", num_return_sequences)

temperature=None
_maybe_add(args, "temperature", temperature)

top_k=None
_maybe_add(args, "top_k", top_k)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(**args)

skip_special_tokens=True

output = tokenizer.decode(
    sample_output[0],
    skip_special_tokens=skip_special_tokens,
)
with open("output.txt", "w") as f:
    f.write(output)
print("\nOutput:\n" + 100 * '-')
print(output)