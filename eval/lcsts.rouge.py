# %%
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import glob
import sacrebleu
import evaluate
import os
import sys

# %%
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
module = evaluate.load(os.path.join(script_dir, "export", "rouge.py"))

# %%


def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    results = []
    for i, x in enumerate(tqdm(lines)):
        output, prediction = x["output"], x["prediction"]
        results.append(module.compute(
            predictions=[prediction],
            references=[output],
            tokenizer=sacrebleu.metrics.bleu._get_tokenizer("zh")()
        ))
    for metric in ["rouge1", "rouge2", "rougeL"]:
        filename = f"{file}.all.{metric}"
        with open(f"{filename}.log", "w+", encoding="utf-8") as log:
            log.writelines([f"{r[metric]}\n" for r in results])

# %%
files = [file for file in sorted(glob.glob("model/*/test/lcsts*.jsonl")) if len(open(file).readlines()) == 8685]
# print(len(files))
process_map(eval_file, files)

# %%
