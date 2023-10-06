# %%
from bleurt import score
import pandas as pd
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import glob
import os
from time import sleep
from multiprocessing.pool import Pool
import os
import sys
os.environ["HF_HOME"] = "/home/user/.cache/huggingface"

# %%
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
scorer = score.BleurtScorer(os.path.join(script_dir, "export/bleurt/model/BLEURT-20"))


# %%
def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    
    model_output  = scorer.score(references=[x["output"] for x in lines], candidates=[x["prediction"] for x in lines], batch_size=64)
    filename = f"{file}.all.bleurt"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        log.writelines([f"{score}\n" for score in model_output])
    

# %%
files = sys.argv[1:]

for file in tqdm(list(files)):
    eval_file(file)
