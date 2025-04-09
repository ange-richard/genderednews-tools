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
"""RCV2"""
from __future__ import absolute_import, division, print_function

import logging

import datasets
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dataset_utils import load_jsonl_dataset


_DESCRIPTION = """RCV2 is a dataset created for the paper """""

_URL = ""
_URLS = {
    "train": _URL + "en_train.jsonl",
    "dev": _URL + "en_val.jsonl",
    "test": _URL + "en_test.jsonl",
}

TAG2ENTITY = {
    '<per>': 'Agent',
    '<peop>': 'Group_of_people',
    '<pron>': 'Source_pronoun',
    '<org>': 'Organization',
    '<cue>': 'Cue',
    '<dirquot>': 'Direct_Quotation',
    '<indquot>': 'Indirect_Quotation',
    '<mixquot>': 'Mixed_Quotation'
    }
ENTITY2TAG = {v:k for k,v in TAG2ENTITY.items()}

REL_LABEL2TEXT = {
    'Quoted_in': 'cité',
    'Indicates': 'indique',
    #'Refers': 'référence'
}
REL_TEXT2LABEL = {v:k for k, v in REL_LABEL2TEXT.items()}

FR_REL_LABEL2TEXT = REL_LABEL2TEXT
FR_REL_TEXT2LABEL = REL_TEXT2LABEL

class RCV2Config(datasets.BuilderConfig):
    """BuilderConfig for RCV2."""

    def __init__(self, **kwargs):
        """BuilderConfig for RCV2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RCV2Config, self).__init__(**kwargs)


class RCV2(datasets.GeneratorBasedBuilder):
    """RCV2 1.0"""

    BUILDER_CONFIGS = [
        RCV2Config(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file_id": datasets.Value("int32"),
                    "jsonl_idx": datasets.Value("int32"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        # this function's **kwargs is a dict of arguments forwarded from the SplitGenerator.gen_kwargs
        """This function returns the examples in the raw (text) form (linearized triplets)."""
        logging.info("generating examples from = %s", filepath)

        dataset = load_jsonl_dataset(filepath[0])
        for idx, row in enumerate(dataset):
            yield lin_example_no_rep_quotes(idx, row)

def lin_example_no_rep_quotes(idx, document):
    entities = document['entities']
    for e in entities: # sort the spans in case they are not
        e['char_span'].sort(key=lambda x: x[0])

    # a relation == [int, str, int, int] -> [relation_id, relation_label, ent1_id, ent2_id]
    quote_relations = [rel for rel in document['relations'] \
                        if rel[1] == 'Quoted_in' or rel[1] == 'Indicates']
    refers_relations = [rel for rel in document['relations'] if rel[1] == 'Refers']

    for rel in quote_relations:
        rel.extend([
            next((e['char_span'][0][0] for e in entities if e['id'] == rel[2]), None), # start char idx of the 1st entity in the rel
            next((e['char_span'][0][0] for e in entities if e['id'] == rel[3]), None), # start char idx of the 2nd entity in the rel
        ])
    quote_relations.sort(key=lambda x: (x[5], x[4]))
    # TODO: sanity check that quote ids are always the second entity in a relation

    # create linearized triplets
    lin_triplets = []

    unique_quote_ids = list(set([rel[3] for rel in quote_relations])) # order is kept
    for q_id in unique_quote_ids:
        # build the linearized triplet for the quote
        rels_to_q_id = []
        # rels_to_q_id are the relations that have quote with id q_id as 2nd entity
        # rels_to_q_id should generally contain two relations, one "Quoted_in" and one "Indicates"
        # however, the "Indicates" rel might be missing (no cue in the text for a given quote)
        for rel in quote_relations: # we loop so that we extract the rels in order
            if rel[3] == q_id:
                rels_to_q_id.append(rel)

        quote = next((e for e in entities if e['id'] == q_id), None)
        triplet_text = f"<triplet>{quote['text']:1}"

        for rel in rels_to_q_id:
            ent = next((e for e in entities if e['id'] == rel[2]), None)
            triplet_text += f"{ENTITY2TAG[quote['label']]:1}{ent['text']:1}{ENTITY2TAG[ent['label']]:1} {REL_LABEL2TEXT[rel[1]]:1}"
        lin_triplets.append(triplet_text)

        # build the linearized triplet for the subject
        rels_to_s_id = []
        # rels_to_s_id are the Refers relations that have subj with id s_id as 2nd entity
        # having more than 1 element in the list is possible, throughout an article
        # the same subj might be referred to multiple times using a pronoun
        subj_id = next((r[3] for r in quote_relations if r[1] == 'Quoted_in'), None)
        if subj_id: # it's rare, but there might be no annotated subject
            for rel in refers_relations:
                if rel[3] == subj_id:
                    rels_to_s_id.append(rel)

            if rels_to_s_id:
                sub = next((e for e in entities if e['id'] == subj_id), None)
                triplet_text = f"<triplet>{sub['text']:1}"

                for rel in rels_to_s_id:
                    ent = next((e for e in entities if e['id'] == rel[2]), None)
                    triplet_text += f"{ENTITY2TAG[sub['label']]:1}{ent['text']:1}{ENTITY2TAG[ent['label']]:1} {REL_LABEL2TEXT[rel[1]]:1}"
                lin_triplets.append(triplet_text)

        # end of triplet building for this quote
    # end of triplet building for this doc
    # print(' '.join(lin_triplets))

    return str(document['id']) + '-' + str(idx), {
        'file_id': document['id'],
        'jsonl_idx': idx,
        'context': document['text'],
        'triplets': ' '.join(lin_triplets),
    }
