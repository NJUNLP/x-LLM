from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, List
from datasets import load_dataset
import torch
import json
import transformers
from transformers import GenerationConfig
import os
import re
import copy

from train import smart_tokenizer_and_embedding_resize, \
	DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
	PROMPT_DICT, \
    DataArguments

import train


@dataclass
class ModelArguments(train.ModelArguments):
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    torch_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )


@dataclass
class GeneratingArguments:
    batch_size: int = field(default=8)
    output_file: str = field(default=None, metadata={"help": "Path to the output."})
    temperature: float = field(default=0.7)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.75)
    top_k: float = field(default=40)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=512)
    template: str = field(default="alpaca")
    labels: Optional[List[str]] = field(default=None)
    transcot: bool = field(default=False)
    transcot_skip_example: bool = field(default=False)
    evaluate: str = field(default="generate")


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GeneratingArguments))
    model_args, data_args, generating_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        torch_dtype=model_args.torch_dtype,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.device_count() > 1:
        from accelerate import load_checkpoint_and_dispatch
        load_checkpoint_and_dispatch(
            model,
            model_args.model_name_or_path,
            device_map="auto",
            offload_state_dict=True,
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.padding_side = "left"

    data_path_base, data_path_name = data_args.data_path.rsplit(os.path.sep, maxsplit=1)
    dataset_name, dataset_config = data_path_name.split("_", maxsplit=1)
    test_dataset = load_dataset(os.path.join(data_path_base, dataset_name), config=dataset_config, split="test")

    def generate_prompt(instruction, input=None, template="alpaca"):
        if template == "alpaca":
            if input:
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        elif template == "raw":
            if input:
                return f"{instruction}\n\n{input}"
            else:
                return f"{instruction}"
        else:
            raise NotImplementedError
        
    def evaluate_by_generate(
        dataset,
        template,
        generation_config
    ):
        prompt = [generate_prompt(ins, inp, template) for ins, inp in zip(dataset["instruction"], dataset["input"])]
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], 
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        return dataset | {"prediction": [o[len(p):].strip() for p, o in zip(prompt, output)]}
    
    def evaluate_by_perplexity(
        dataset,
        template,
        labels
    ):
        label_perplexity = []
        for label in labels:
            prompt = [generate_prompt(ins, inp, template) + label for ins, inp in zip(dataset["instruction"], dataset["input"])]
            inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                ).logits
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_targets = inputs["input_ids"][..., 1:].contiguous()
            shift_attention_mask_batch = inputs["attention_mask"][..., 1:].contiguous()
            perplexity = torch.exp(
                (torch.nn.CrossEntropyLoss(reduction="none")(shift_logits.transpose(1, 2), shift_targets) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            label_perplexity.append(perplexity)
        prediction = [labels[l] for l in torch.stack(label_perplexity).argmin(dim=0).detach().cpu()]
        return dataset | {"prediction": prediction}
    
    
    generation_config = GenerationConfig(
        temperature=generating_args.temperature,
        do_sample=generating_args.do_sample,
        top_p=generating_args.top_p,
        top_k=generating_args.top_k,
        num_beams=max(2, generating_args.num_beams) if generating_args.labels else generating_args.num_beams,
        max_new_tokens=generating_args.max_new_tokens,
        force_word_ids=[tokenizer(generating_args.labels, add_special_tokens=False)["input_ids"]] if generating_args.labels else None
    )
    
    if generating_args.transcot:
        translation_cache = dict()
        translation_generation_config = GenerationConfig(
            temperature=generating_args.temperature,
            do_sample=generating_args.do_sample,
            top_p=generating_args.top_p,
            top_k=generating_args.top_k,
            num_beams=generating_args.num_beams,
            max_new_tokens=512,
        )
    
    with open(generating_args.output_file, "w") as output_file:
        for i in tqdm(range(0, len(test_dataset), generating_args.batch_size)):
            d = test_dataset[i:i + generating_args.batch_size]
            
            ## ? translate input
            if generating_args.transcot:
                d["original_input"] = copy.deepcopy(d["input"])
                _DELIM = "\n"
                _EXAMPLE_DELIM = "\n\n"
                trans_dataset = {
                    "input": [],
                    "sample_id": [],
                    "line_id": [],
                    "trans_input": [],
                }
                ### build translation dataset
                for sample_id, sample in enumerate(d["input"]):
                    if generating_args.transcot_skip_example:
                        lines_to_translate = sample.split(_EXAMPLE_DELIM)[-1].split(_DELIM)
                    else:
                        lines_to_translate = sample.split(_DELIM)
                    for line_id, line in enumerate(lines_to_translate):
                        if re.match(r"[A-D]\. ", line) is not None:
                            line = line[3:]
                        if line.strip() and line not in translation_cache.keys() and not line.startswith("Answer:"):
                            trans_dataset["input"].append(line)
                            trans_dataset["sample_id"].append(sample_id)
                            trans_dataset["line_id"].append(line_id)
                ### run translation
                for i in range(0, len(trans_dataset["input"]), generating_args.batch_size):
                    td = trans_dataset["input"][i:i + generating_args.batch_size]
                    trans_output = evaluate_by_generate({
                            "input": td,
                            "instruction": ["Translate the following sentences to English."] * len(td),
                        }, 
                        template="alpaca",
                        generation_config=translation_generation_config
                    )
                    trans_dataset["trans_input"] += trans_output["prediction"]
                    for inp, pre in zip(trans_output["input"], trans_output["prediction"]):
                        translation_cache[inp] = pre
                ### change input
                d["input"] = [
                    _DELIM.join([
                        (translation_cache[line] if line in translation_cache.keys() else line)
                        if re.match(r"[A-D]\. ", line) is None
                        else (line[:3] + translation_cache[line[3:]] if line[3:] in translation_cache.keys() else line)
                        for line in sample.split(_DELIM)
                    ])
                    for sample in d["original_input"]
                ]
            ## ? translate input

            if generating_args.evaluate == "generate":
                output = evaluate_by_generate(d, template=generating_args.template, generation_config=generation_config)
            elif generating_args.evaluate == "perplexity":
                assert generating_args.labels, "evaluate with perplexity requires labels"
                output = evaluate_by_perplexity(d, template=generating_args.template, labels=generating_args.labels)
            output_file.writelines(
                json.dumps(sample, ensure_ascii=False) + "\n" for sample in [dict(zip(output.keys(),t)) for t in zip(*output.values())]
            )
            output_file.flush()
    
if __name__ == "__main__":
    inference()
    