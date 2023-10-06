# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import os
import json
import datasets
logger = datasets.logging.get_logger(__name__)

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_LANG = ["ar", "el", "en", "es", "hi", "ru", "tr", "vi", "zh"]

class MIEvalConfig(datasets.BuilderConfig):

    """BuilderConfig for MIEval"""

    def __init__(self, config: str, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(MIEvalConfig, self).__init__(**kwargs)
        self.lang = config


class MIEval(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [MIEvalConfig(config=lang, name=f"mieval_{lang}") for lang in _LANG]
    BUILDER_CONFIG_CLASS = MIEvalConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            ),
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(self.base_path, f"{self.config.lang}.jsonl")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("[mieval] generating examples from = %s", filepath)
        
        with open(filepath, encoding="utf-8") as f:
            mieval = [json.loads(x) for x in f.readlines()]
            id_ = 0
            for sample in mieval:
                yield id_, sample | {"id": id_, "output": ""}
                id_ += 1
