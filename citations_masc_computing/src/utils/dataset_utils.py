# preprocess the samples and create datasets as needed

import os
import json
import logging
import argparse

from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SEED = 7
DEV_TEST_TRAIN_RATIO = [0.15, 0.15, 0.70]

REL_LABEL2ID = {
    'Quoted_in': 1,
    'Indicates': 2,
    'Refers': 3
}
REL_ID2LABEL = {v: k for k, v in REL_LABEL2ID.items()}

ENT_LABEL2ID = {
    'Agent': 1,
    'Group_of_people': 2,
    'Source_pronoun': 3,
    'Organization': 4,
    'Cue': 5,
    'Direct_Quotation': 6,
    'Indirect_Quotation': 7,
    'Mixed_Quotation': 8
}
ENT_ID2LABEL = {v: k for k, v in ENT_LABEL2ID.items()}


def load_jsonl_dataset(jsonl_file):
    data = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_json_dataset_from_files(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if os.path.splitext(filename)[0].isnumeric() and filename.endswith('.json'):
            with open(os.path.join(data_dir, filename)) as f:
                data.append(json.load(f))
    return data


def map_json_labels2ids(dataset, map_entities=True, map_relations=True):
    # Replace ent and relation labels by ids in all files
    for item in dataset:
        if map_entities:
            new_entities = []
            for ent in item['entities']:
                new_ent = ent
                new_ent['label'] = ENT_LABEL2ID[new_ent['label']]
                new_entities.append(new_ent)
            item['entities'] = new_entities

        if map_relations:
            new_relations = []
            for rel in item['relations']:
                new_relations.append([rel[0], REL_LABEL2ID[rel[1]], rel[2], rel[3]])
            new_relations.sort(key=lambda x: x[0])
            item['relations'] = new_relations
    return dataset


def map_json_ids2labels(dataset, map_entities=True, map_relations=True):
    # Replace ent and relation ids by labels in all files
    for item in dataset:
        if map_entities:
            new_entities = []
            for ent in item['entities']:
                new_ent = ent
                new_ent['label'] = ENT_ID2LABEL[new_ent['label']]
                new_entities.append(new_ent)
            item['entities'] = new_entities

        if map_relations:
            new_relations = []
            for rel in item['relations']:
                new_relations.append([rel[0], REL_ID2LABEL[rel[1]], rel[2], rel[3]])
            new_relations.sort(key=lambda x: x[0])
            item['relations'] = new_relations
    return dataset


def split_docs_to_tok_len_and_save_for_prediction(
        data_dir,
        from_jsonl=False,
        save=False,
        output_dir=None,  # path without filename
        max_tok_len=512):
    # Read a dataset from files or a .jsonl, extend the dataset either return it as a [{}] or save it to a .jsonl
    # This function takes and returns files where "label" values are text labels and not id labels
    # We switch to id labels before tokenization (to avoid pyarrow tokenization error)
    # and return to text labels before writing out manually (pyarrow writer cannot write text labels, only ids)

    dataset = load_jsonl_dataset(data_dir) if from_jsonl else load_json_dataset_from_files(data_dir)
    print(f"Total number of documents: {len(dataset)}")
    dataset = Dataset.from_list(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        'Babelscape/rebel-large')  # we use rebel's (i.e. BART) tokenizer to compute where to make the cuts since in BARThez the newline token is not defined

    def tokenize(example):
        return tokenizer(
            example['text'],
            max_length=max_tok_len,
            truncation=True,
            return_length=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=False)

    def split_into_chunks(example):
        # remove ids and offset related to tokenizer special tokens and create flat lists
        #print(f"Processing {example['id']}.json")
        ids = []
        offsets = []
        for i, m in zip(example['input_ids'], example['offset_mapping']):
            ids.extend(i[1:-1])
            offsets.extend(m[1:-1])

        # compute char cuts
        newline_id = tokenizer.encode('\n', add_special_tokens=False)[0]
        fullstop_id = tokenizer.encode('.', add_special_tokens=False)[0]
        quote_stop_id = tokenizer.encode('".', add_special_tokens=False)[0]
        end_sign_id = tokenizer.encode('/.', add_special_tokens=False)[0]

        char_breaks = []
        while ids and offsets:  # compute text chunks
            max_tok = len(ids) if len(ids) <= max_tok_len else max_tok_len
            last_newline_tok_idx = None
            # get the offset of the closest \n to the left of the 512 token mark
            for idx, input_id in reversed(list(enumerate(ids[:max_tok]))):
                if input_id == newline_id:
                    last_newline_tok_idx = idx
                    break
            # in case we have no \n in the text: stop at last . or ". (which may match other punctuation so not ideal)
            if not last_newline_tok_idx:
                for idx, input_id in reversed(list(enumerate(ids[:max_tok]))):
                    if input_id in [end_sign_id,fullstop_id,quote_stop_id]:
                        last_newline_tok_idx = idx
                        break

            assert last_newline_tok_idx, f"No newline nor full stop in example {example['id']}"
            # TODO: fix problem! when last newline char is ". -> is cut in two in extended dataset.
            last_newline_char_idx = offsets[last_newline_tok_idx][0]  # offsets end bounds are non-inclusive
            char_breaks.append(last_newline_char_idx)

            ids = ids[last_newline_tok_idx + 1:]
            offsets = offsets[last_newline_tok_idx + 1:]

        # compute char cut spans
        cut_char_spans = []
        for i, c in enumerate(char_breaks):
            if i == 0:
                cut_char_spans.append([0, c + 1])  # make end bound non-inclusive
            else:
                next_start_char_idx = cut_char_spans[-1][1]
                next_end_char_idx = next_start_char_idx + c
                cut_char_spans.append([next_start_char_idx, next_end_char_idx])

        return cut_char_spans

    # actual processing starts here
    tokenized_dataset = dataset.map(tokenize, desc='Running tokenization on dataset')

    extended_dataset = []  # will be a [{}]
    example_counter = 0
    no_endpunct_counter = 0
    extended_ids = []
    # build extended dataset from chunks
    for example in tokenized_dataset:
        logger.info(f'Splitting examples with texts longer than {max_tok_len} tokens')
        if len(example['length']) == 1:
            extended_dataset.append(example)
            example_counter += 1
            extended_ids.append(example["id"])
        else:
            # create new examples from the chunks
            try:
                example_counter += 1
                chunk_spans = split_into_chunks(example)
                for chunk in chunk_spans:
                    extended_dataset.append({
                        'id': example['id'],
                        'text': example['text'][chunk[0]:chunk[1]]
                    })
                    extended_ids.append(example["id"])
            except AssertionError:
                no_endpunct_counter += 1
                logger.info(f"Skipping example {example['id']}.")

    # remove unnecessary keys from tokenization and save
    keys_to_keep = ['id', 'text']
    extended_dataset = [{k: item[k] for k in keys_to_keep} for item in extended_dataset]
    print(f"Number of unique ids: {len(list({v['id']: v for v in extended_dataset}.values()))}")
    assert example_counter == len(tokenized_dataset), \
        f"Dataset extension build size mismatch, counter:{example_counter}, tokenized {len(tokenized_dataset)}"
    logger.info(f'{no_endpunct_counter} example(s) discarded for formating issues.')

    if save:
        filename = data_dir.split("/")[-1].split(".")[0]
        outfile = f'{filename}_extended_dataset.jsonl'
        output_dir = data_dir if not output_dir else output_dir
        logger.info(f'Saving extended dataset to {os.path.join(output_dir, outfile)}')
        with open(os.path.join(output_dir, outfile), 'w') as jsonl_file:
            for item in extended_dataset:
                json.dump(item, jsonl_file, ensure_ascii=False)
                jsonl_file.write('\n')
    else:
        return extended_dataset

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to directory containing a .json or .jsonl dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to directory to write output files')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Name or the .json or .jsonl dataset to be generated')
    args = parser.parse_args()

    split_docs_to_tok_len_and_save_for_prediction(
        args.input_dir,
        from_jsonl=True,
        save=True,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()