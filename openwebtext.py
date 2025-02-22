# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""The Open WebText Corpus"""

import re

import datasets


_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
An open-source replication of the WebText dataset from OpenAI.
"""

_N_DATA_FILES = 21
_DATA_FILES = ["subsets/urlsf_subset{:02d}.tar".format(i) for i in range(_N_DATA_FILES)]


class Openwebtext(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archives = dl_manager.download(_DATA_FILES)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "archive_iterators": [
                    dl_manager.iter_archive(archive) for archive in archives
                ],
                "iter_archive": dl_manager.iter_archive
            }),
        ]

    def _generate_examples(self, archive_iterators, iter_archive):
        """Yields examples."""
        for archive_iterator in archive_iterators:
            for xz_filepath, xz_f in archive_iterator:
                if not xz_filepath.endswith(".xz"):
                    continue
                for txt_filepath, txt_f in iter_archive(xz_f):
                    if not txt_filepath.endswith(".txt"):
                        continue
                    idx = f"{xz_filepath}/{txt_filepath}"
                    yield idx, {"text": re.sub("\n\n\n+", "\n\n", txt_f.read().decode("utf-8")).strip()}