
import argparse
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from datasets import Dataset
from transformers import AutoTokenizer

#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_builders import rcv2 as rcv2_dataset
from utils import dataset_utils


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class Trilinear(nn.Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        in3_features: size of each third input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = Trilinear(20, 30, 40, 50)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> input3 = torch.randn(128, 40)
        >>> output = m(input1, input2, input3)
        >>> print(output.size())
        torch.Size([128, 50])
    """
    __constants__ = ['in1_features', 'in2_features', 'in3_features', 'out_features']
    in1_features: int
    in2_features: int
    in3_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, in3_features: int, out_features: int, bias: bool = True) -> None:
        super(Trilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.in3_features = in3_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features, in3_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
        if self.bias is not None:
            return torch.einsum('bn,bm,bo,anmo->ba', input1, input2, input3, self.weight) + self.bias
        return torch.einsum('bn,bm,bo,anmo->ba', input1, input2, input3, self.weight)
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, in3_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.in2_features, self.out_features, self.bias is not None
        )

class BartTripletHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense_head_tail_ctxt = nn.Linear(input_dim*3, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, head_states: torch.Tensor, tail_states: torch.Tensor, context_states: torch.Tensor):
        combined_state = torch.cat((head_states, tail_states, context_states), dim = 1)
        combined_state = self.dropout(combined_state)
        hidden_states = self.dense_head_tail_ctxt(combined_state)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

    return shifted_input_ids

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

def extract_triplets_typed(text, mapping_types={'<peop>': 'Peop', '<org>': 'Org', '<other>': 'Other', '<loc>': 'Loc'}):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = mapping_types[token]
            else:
                current = 'o'
                object_type = mapping_types[token]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets


def extract_triplets_rcv2(text, mapping_types=None, build_data_for_mREBEL=False):
    # print(f"============ TEXT TO EXTRACT TRIPLETS FROM =============\n {text}")
    mapping_types = rcv2_dataset.TAG2ENTITY if not mapping_types else mapping_types
    REL_LABEL2TEXT = rcv2_dataset.EN_REL_LABEL2TEXT if build_data_for_mREBEL else rcv2_dataset.FR_REL_LABEL2TEXT

    def extract_relations(lst):
        # cut triplets based on the appearance of relation labels
        rels = []
        rel_items = []
        for item in (lst):
            rel_items.append(item)

            if item in REL_LABEL2TEXT.values():
                rels.append(rel_items)
                rel_items = []
        return rels

    # warn: special tokens are "pasted" to the preceding word by the tokenizer, so we add back the missing whitespace
    text = text.strip()
    word_list = text.replace('fr_XX', '').replace('tp_XX', '').replace('<', ' <').replace('<s>', '').replace('</s>',
                                                                                                             '').replace(
        '<pad>', '').split()
    if not word_list:  # some texts might be only a sequence of special characters so the word_list will be empty
        return []
    # print(f"text: {text}")
    # print(f"cleaned word list: {word_list}")

    # at the beggining of training, there might not be any or not enough tags/items in the prediction
    # to extract, so we make sure to handle these cases
    triplet_indices = [idx for idx, word in enumerate(word_list) if word == '<triplet>']
    # print(f"triplet indices: {triplet_indices}")

    triplet_list = []  # [[]] where each [] contains all the triplets from a doc word by word
    if triplet_indices:
        for item in reversed(triplet_indices):
            triplet_list.append(word_list[item:])
            word_list = word_list[:item]
        triplet_list = [t[1:] for t in triplet_list]  # remove <triplet> tag from each list
        triplet_list = list(reversed(triplet_list))
        # triplet_list = [' '.join(t) for t in triplet_list]
        # print(triplet_list)
    else:
        # no triplet tags were found, we assign the whole predicted text as triplet
        triplet_list.append(word_list)
    # print(f"triplet list: {triplet_list}")

    triplets = []  # [[]] where each [] contains the triplets item by item (either tags or full sentences)
    for t in triplet_list:
        items = []
        text = ''
        for idx, word in enumerate(t):
            if word not in mapping_types.keys() and idx != len(t) - 1:
                text += f'{word} '
            elif word not in mapping_types.keys() and idx == len(t) - 1:  # last word of the triplet
                text += f'{word} '
                items.append(text)
            elif word in mapping_types.keys() and text != '':
                items.extend([text, word])
                text = ''
        items = [w.strip() for w in items]
        triplets.append(items)
    # print(f"triplets: {triplets}")

    # we return the relations in the direction they were annotated (refer to .jsonl dataset files)
    # refer to paper section _ for details on the triplet structure extracted here
    relations = []

    # TODO: check what to do if empty triplets are generated
    if triplets:
        for triplet in triplets:
            # check if the correct number of tags/items per triplet were predicted, return empty strings if needed

            if len(triplet) > 1:
                tail = triplet[0]  # the tail will always be either a quote or a subject
                rels = extract_relations(triplet[1:])

                for rel in rels:
                    if len(rel) == 4:
                        relations.append(
                            {'head': rel[1], 'head_type': rel[2], 'tail': tail, 'tail_type': rel[0], 'type': rel[3]})
                    else:
                        relation = {'head': None, 'head_type': None, 'tail': tail, 'tail_type': None, 'type': None}
                        for item in rel:
                            if item in list(REL_LABEL2TEXT.values())[:3] or item in list(REL_LABEL2TEXT.values())[
                                                                                    5:] and not relation['tail_type']:
                                relation['tail_type'] = item
                            elif item in list(mapping_types.keys()) and not relation['head_type']:
                                relation['head_type'] = item
                            elif item in REL_LABEL2TEXT.values() and not relation['type']:
                                relation['type'] = item
                            elif item not in list(mapping_types.keys()) and not relation['head']:
                                relation['head'] = item

                        for k, v in relation.items():
                            if not v:
                                relation[k] = ''
                        relations.append(relation)

            elif len(triplet) == 1:
                # in the beg of training, BARThez REBEL sometimes predicted a single chunk of text with no labels
                relations.append({'head': '', 'head_type': '', 'tail': triplet[0], 'tail_type': '', 'type': ''})

    # print(f"relations: {relations}")
    return relations


def count_max_token_pieces(path_to_jsonl):
    """ Linearize a dataset and compute the max token length """
    dataset = dataset_utils.load_jsonl_dataset(path_to_jsonl)
    linearized_dataset = []
    for idx, row in enumerate(dataset):
        _, example_dict = rcv2_dataset.lin_example_no_rep_quotes(idx, row)
        linearized_dataset.append({'text': example_dict['triplets']})
    dataset = Dataset.from_list(linearized_dataset)

    tokenizer = AutoTokenizer.from_pretrained('moussaKam/barthez')
    tokenizer.add_tokens([
        '<triplet>', '<per>', '<peop>', '<pron>', '<org>', '<cue>', '<dirquot>', '<indquot>', '<mixquot>'], special_tokens = True)
    def tokenize(example):
        return tokenizer(example['text'], return_length=True, return_attention_mask=False)
    tokenized_dataset = dataset.map(tokenize, desc='Running tokenization on dataset')
    print(f"Max token length of the dataset: {max(tokenized_dataset['length'])}")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to directory containing jsonl file.')
    args = parser.parse_args()

    count_max_token_pieces(args.input_dir)


if __name__ == '__main__':
    main()