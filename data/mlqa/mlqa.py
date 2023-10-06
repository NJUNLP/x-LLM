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

# THIS DATASET USES LANGUAGE SPECIFIC INSTRUCTION

_INSTRUCTIONS = {
"ar": "الرجاء الإجابة على الأسئلة بناءً على الفقرات التالية",
"en": "Answer the final question with following context",
"hi": "कृपया निम्नलिखित पैराग्राफ के अनुसार प्रश्नों के उत्तर दें",
"vi": "Hãy trả lời các câu hỏi theo đoạn văn sau",
"zh": "请根据以下段落，回答问题",
}

_TEMPLATES = {
"ar": """\
فقرة: {context}
سؤال: {question}
إجابة: """,

"en": """\
Context: {context}
Question: {question}
Answer: """,

"hi": """\
अनुच्छेद: {context}
सवाल: {question}
उत्तर: """,

"vi": """\
đoạn văn: {context}
câu hỏi: {question}
Trả lời: """,

"zh":"""\
段落: {context}
问题: {question}
答案: """,
}

_CITATION = """\
@article{lewis2019mlqa,
  title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
  author={Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
  journal={arXiv preprint arXiv:1910.07475},
  year={2019}
}
"""

_DESCRIPTION = """\
MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
4 different languages on average.
"""

_LANG = ["ar", "de", "zh", "vi", "en", "es", "hi"]

class MLQAConfig(datasets.BuilderConfig):

    """BuilderConfig for MLQA"""

    def __init__(self, config: str, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(MLQAConfig, self).__init__(**kwargs)
        self.lang = config


class MLQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [MLQAConfig(config=lang, name=f"mlqa_{lang}") for lang in _LANG]
    BUILDER_CONFIG_CLASS = MLQAConfig

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
            homepage="https://github.com/facebookresearch/MLQA",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(self.base_path, f"test-context-{self.config.lang}-question-{self.config.lang}.json")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("[mlqa] generating examples from = %s", filepath)
        
        with open(filepath, encoding="utf-8") as f:
            mlqa = json.load(f)
            id_ = 0
            for article in mlqa["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "id": qa["id"],
                            "instruction": _INSTRUCTIONS[self.config.lang],
                            "input": _TEMPLATES[self.config.lang].format_map({
                                "context": context,
                                "question": question,
                            }),
                            "output": answers[0]
                        }
                        id_ += 1
