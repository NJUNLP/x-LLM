# %%
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import glob
import sys
from sacrebleu.metrics import BLEU
bleu = BLEU(tokenize="flores101", effective_order=True)

# %%
def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    cases = list(map(
        lambda x: (x["output"], x["prediction"]),
        lines
    ))
    filename = f"{file}.all.bleu_sent"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        for i, case in enumerate(tqdm(cases)):
            output, prediction = case
            log.write("{}\n".format(bleu.sentence_score(
                hypothesis=prediction,
                references=[output]
            ).score))

    filename = f"{file}.all.bleu"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        log.write("{}\n".format(bleu.corpus_score(
            hypotheses=[x["prediction"] for x in lines],
            references=[[x["output"] for x in lines]]
        ).score))

# %%
files = sys.argv[1:]
process_map(eval_file, files)
