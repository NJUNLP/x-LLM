# %%
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import glob
import sys
import evaluate
import os

# %%
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
module = evaluate.load(os.path.join(script_dir, "export", "sari.py"))

# %%


def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    filename = f"{file}.all.sari"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        for i, x in enumerate(tqdm(lines)):
            input, output, prediction = x["input"], x["output"], x["prediction"]
            output = eval(output)
            log.write("{}\n".format(module.compute(
                sources=[input],
                predictions=[prediction],
                references=[output],
                tokenizer="zh"
            )["sari"]))


# %%
files = sys.argv[1:]
process_map(eval_file, files)
