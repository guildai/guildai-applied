#Using BLEU score to compare the real sentences with the generated ones
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import statistics


test_set = pd.read_pickle("generated_set.pkl")
true_endings = test_set['True_end_lyrics']
generated_endings = test_set['Generated_lyrics']
scores=[]
for i in range(len(test_set)):
  reference = true_endings[i].split()
  candidate = generated_endings[i].split(".")[0].split()
  scores.append(
    sentence_bleu(
      [reference],
      candidate,
    )
  )

avg_bleu = statistics.mean(scores)
print(f"avg_bleu: {avg_bleu}")