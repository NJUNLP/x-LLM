# %%
import comet  # From: unbabel-comet
import pandas as pd
import torch
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
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
scorer = comet.load_from_checkpoint("/home/user/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt")

# %%
TOTAL_GPUS = 1
def eval_file(args):
    i, file = args
    print(f"[start evaluation]: all in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    
    data = [{
        "src": sample["input"], 
        "mt": sample["prediction"], 
        "ref": sample["output"]
    } for sample in lines]
    model_output  = scorer.predict(data, gpus=1 if torch.cuda.is_available() else 0, devices=[i % TOTAL_GPUS], accelerator="cuda", batch_size=256, num_workers=0)
    filename = f"{file}.all.comet"
    with open(f"{filename}.log", "w+", encoding="utf-8") as log:
        log.writelines([f"{score}\n" for score in model_output.scores])
    

# %%
files = sys.argv[1:]

with Pool(processes=TOTAL_GPUS) as pool:
    for result in pool.imap(eval_file, list(enumerate(files))):
        continue
