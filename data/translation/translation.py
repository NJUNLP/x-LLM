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

from iso639 import languages
import os
import re
import datasets
logger = datasets.logging.get_logger(__name__)

_INSTRUCTIONS = [
    "Translate the following sentences from {source_lang} to {target_lang}.", 
    "What do the following sentences mean in {target_lang}?", 
    "Please provide the {target_lang} translation for the following sentences.", 
    "Convert the subsequent sentences from {source_lang} into {target_lang}.", 
    "Render the listed sentences in {target_lang} from their original {source_lang} form.", 
    "Transform the upcoming sentences from {source_lang} language to {target_lang} language.", 
    "Change the given sentences from {source_lang} to {target_lang} format.", 
    "Turn the following sentences from their {source_lang} version to the {target_lang} version.", 
    "Adapt the mentioned sentences from {source_lang} to the {target_lang} language.", 
    "Transpose the next sentences from the {source_lang} format to the {target_lang} format.", 
    "Switch the specified sentences from their {source_lang} form to {target_lang} form.", 
    "Reinterpret the ensuing sentences from {source_lang} to {target_lang} language.", 
    "Modify the forthcoming sentences, converting them from {source_lang} to {target_lang}.", 
    "How can the subsequent sentences be interpreted in {target_lang}?", 
    "What is the meaning of these sentences when translated to {target_lang}?", 
    "In the context of {target_lang}, what do the upcoming sentences signify?", 
    "How would you express the meaning of the following sentences in {target_lang}?", 
    "What is the significance of the mentioned sentences in {target_lang}?", 
    "In {target_lang}, what do the given sentences convey?", 
    "When translated to {target_lang}, what message do these sentences carry?", 
    "What is the intended meaning of the ensuing sentences in {target_lang}?", 
    "How should the following sentences be comprehended in {target_lang}?", 
    "In terms of {target_lang}, what do the next sentences imply?", 
    "Kindly furnish the {target_lang} translation of the subsequent sentences.", 
    "Could you supply the {target_lang} translation for the upcoming sentences?", 
    "Please offer the {target_lang} rendition for the following statements.", 
    "I'd appreciate it if you could present the {target_lang} translation for these sentences.", 
    "Can you deliver the {target_lang} translation for the mentioned sentences?", 
    "Please share the {target_lang} version of the given sentences.", 
    "It would be helpful if you could provide the {target_lang} translation of the ensuing sentences.", 
    "Kindly submit the {target_lang} interpretation for the next sentences.", 
    "Please make available the {target_lang} translation for the listed sentences.", 
    "Can you reveal the {target_lang} translation of the forthcoming sentences?", 
]

_CITATION = """\
This is an adapter for loading raw text parallel corpus.
"""

_DESCRIPTION = """\
This is an adapter for loading raw text parallel corpus.
"""

class TranslationDataConfig(datasets.BuilderConfig):
    """BuilderConfig for TranslationData."""

    def __init__(self, config: str, **kwargs):
        """BuilderConfig for TranslationData.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TranslationDataConfig, self).__init__(**kwargs)
        self.dataset, lang = config.rsplit("_", maxsplit=1)
        self.source_lang, self.target_lang = lang.split("-")


class TranslationData(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    BUILDER_CONFIG_CLASS = TranslationDataConfig

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
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        base_path = os.path.join(self.base_path, self.config.dataset, f"{self.config.source_lang}-{self.config.target_lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(base_path, "test")}),
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(base_path, "train")}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(base_path, "validation")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        source_name, target_name = (languages.get(alpha2=lang).name for lang in [self.config.source_lang, self.config.target_lang])
        key = 0
        if os.path.exists(f"{filepath}.{self.config.source_lang}") and os.path.exists(f"{filepath}.{self.config.target_lang}"):
            with open(f"{filepath}.{self.config.source_lang}", encoding="utf-8") as source_f, open(f"{filepath}.{self.config.target_lang}", encoding="utf-8") as target_f:
                for source_line, target_line in zip(source_f, target_f):
                    if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                        yield key, {
                            "id": key,
                            "instruction": _INSTRUCTIONS[key % len(_INSTRUCTIONS) if "train" in filepath else 0].format_map({
                                "source_lang": source_name, 
                                "target_lang": target_name, 
                            }),
                            "input": source_line.strip(),
                            "output": target_line.strip(),
                        }
                    key += 1
