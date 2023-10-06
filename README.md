# <img src="./llama.png" alt="Icon" width="50" height="50"> Extrapolating Large Language Models to Non-English by Aligning Languages

This repository contains the code implementation for the project that aims to empower pre-trained Large Language Models (LLMs) on non-English languages by building semantic alignment across languages. The project explores cross-lingual instruction-tuning and multilingual instruction-tuning techniques. The code implementation is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Requirements and Installation
To install this repository, follow these steps:
```
git clone git@github.com:NJUNLP/x-LLM.git
cd x-LLM
pip install --editable ./
```

For detailed information about the conda environment, refer to the environment.yml file.

## Usage
### Download Pre-trained LLM
Start by downloading the pre-trained LLM into the ./model directory.

### Download Dataset
You can download all the datasets used in this project from this [link](https://drive.google.com/file/d/1Vk17GBIoJNN0QGqsTLWSuzEMvr4bFTvm/view?usp=drive_link). Once downloaded, place the datasets in the ./data directory. The datasets include:

* Training dataset
  * Alpaca
  * Wikimatrix
  * Newscommentary
* Evaluation dataset
  * XQUAD
  * MLQA
  * Flores-101
  * MI-Eval

### Load Raw Data Along with Instruction 
You can load raw data along with instruction using the provided scripts (./data/<dataset>/<dataset.py>). If you want to use a new dataset, you need to implement the corresponding script. The loaded data will have the following structure:
``` python
datasets.Features(
    {
        "id": datasets.Value("string"),
        "instruction": datasets.Value("string"),
        "input": datasets.Value("string"),
        "output": datasets.Value("string")
    }
)
```

## Instruction-tune Pre-trained LLM
To instruction-tune the pre-trained LLM, run the train.sh script. For example, you can instruction-tune LLaMA-7B to x-LLaMA-7B (Chinese) with the following command:
``` bash
bash script/train.sh llama-7b-hf alpaca_en+alpaca_zh+translation_ncwm_en-zh
```
In this command, the first argument denotes the pre-trained LLM to use, and the second argument represents the training data to use. You can use + to concatenate multiple datasets, and the training data will be shuffled by the Huggingface Trainer.

Once the training is complete, the finetuned LLM will be saved in ./model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune. You can use aliases to define shorter names, and more details can be found in ./data/alias/alias.json.

## Test Finetuned LLM
To test the finetuned LLM, run the inference.sh script. For example, you can test the tuned LLM on the Flores dataset with the following command:
``` bash
bash script/inference.sh llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune translation_flores_en-zh
```
The output results will be saved in model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune/test/translation_flores_en-zh.inference.jsonl. The prediction field represents the generated content of the LLM.

## Interact with LLM Through Web UI
To interact with the LLM through a web UI, run app.py with the following command:
``` bash
bash app.py model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune
```

## Citation
If you find this repository helpful, please consider citing our paper:
```
@misc{zhu2023extrapolating,
      title={Extrapolating Large Language Models to Non-English by Aligning Languages}, 
      author={Wenhao Zhu and Yunzhe Lv and Qingxiu Dong and Fei Yuan and Jingjing Xu and Shujian Huang and Lingpeng Kong and Jiajun Chen and Lei Li},
      year={2023},
      eprint={2308.04948},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```