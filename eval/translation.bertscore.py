# %%
from bert_score import BERTScorer
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
scorer = BERTScorer(model_type=os.path.join(script_dir, "export/bertscore/model/bert-base-multilingual-cased"), num_layers=9)


# %%
def eval_file(file):
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    
    p, r, f1 = scorer.score(refs=[x["output"] for x in lines], cands=[x["prediction"] for x in lines], batch_size=1024)
    filename = f"{file}.all.bertscore"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        log.writelines([f"{score}\n" for score in f1])
    

# %%
files = sys.argv[1:]

for file in tqdm(list(files)):
    eval_file(file)
