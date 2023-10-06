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

# THIS DATASET USES LANGUAGE SPECIFIC INSTRUCTION

import os
import json
import datasets
logger = datasets.logging.get_logger(__name__)

_INSTRUCTIONS = {
"ar": "الرجاء الإجابة على الأسئلة بناءً على الفقرات التالية",
"el": "Απαντήστε στις ερωτήσεις με βάση τις παρακάτω παραγράφους",
"en": "Answer the final question with following context",
"hi": "कृपया निम्नलिखित पैराग्राफ के अनुसार प्रश्नों के उत्तर दें",
"tr": "Lütfen soruları aşağıdaki paragraflara göre cevaplayınız",
"vi": "Hãy trả lời các câu hỏi theo đoạn văn sau",
"zh": "请根据以下段落，回答问题",
}

_TEMPLATES = {
"ar": """\
فقرة: {context}
سؤال: {question}
إجابة: """,

"el": """\
παράγραφος: {context}
ερώτηση: {question}
Απάντηση: """,

"en": """\
Context: {context}
Question: {question}
Answer: """,

"hi": """\
अनुच्छेद: {context}
सवाल: {question}
उत्तर: """,

"tr": """\
paragraf: {context}
soru: {question}
Cevap: """,

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
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
"""

_DESCRIPTION = """\
XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering
performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set
of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German,
Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi and Romanian. Consequently, the dataset is entirely parallel
across 12 languages.
"""

_LANG = ["ar", "de", "zh", "vi", "en", "es", "hi", "el", "th", "tr", "ru", "ro"]

class XquadConfig(datasets.BuilderConfig):

    """BuilderConfig for Xquad"""

    def __init__(self, config: str, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XquadConfig, self).__init__(**kwargs)
        self.lang = config


class Xquad(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [XquadConfig(config=lang, name=f"xquad_{lang}") for lang in _LANG]
    BUILDER_CONFIG_CLASS = XquadConfig

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
            homepage="https://github.com/deepmind/xquad",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(self.base_path, f"xquad.{self.config.lang}.json")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("[xquad] generating examples from = %s", filepath)
        
        with open(filepath, encoding="utf-8") as f:
            xquad = json.load(f)
            id_ = 0
            for article in xquad["data"]:
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
