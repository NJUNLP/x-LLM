# %%
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import glob
import os
import sys

# %%
def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    cases = list(map(
        lambda x: (x["output"], x["prediction"]), 
        lines
    ))
    filename = f"{file}.all.match"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        for i, case in enumerate(tqdm(cases)):
            output, prediction = case
            log.write("{}\n".format("1" if output.lower() in prediction.lower() else "0"))
        
# %%
files = sys.argv[1:]
process_map(eval_file, files)
