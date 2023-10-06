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

# dummy dataset uses load datasets for alias

import os
import json
import datasets
logger = datasets.logging.get_logger(__name__)


class AliasConfig(datasets.BuilderConfig):

    """BuilderConfig for Alias"""

    def __init__(self, config: str, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(AliasConfig, self).__init__(**kwargs)
        self.alias = config


class Alias(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = AliasConfig

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "from": datasets.Value("string"),
                }
            ),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": ""}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("[alias] generating examples from = %s", filepath)
        
        alias = json.load(open(os.path.join(self.base_path, "alias.json")))
        full_dataset = alias[self.config.alias]
        
        import sys
        sys.path.append(os.path.join(os.path.pardir, os.path.pardir))
        from utils import load_datasets
        
        for _id, d in enumerate(load_datasets(os.path.join(self.base_path, os.path.pardir, full_dataset))):
            yield _id, d
