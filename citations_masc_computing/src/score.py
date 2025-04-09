#!/usr/bin/env python3

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""
import sys
import argparse
import logging
import numpy as np

from collections import Counter, namedtuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

Rel = namedtuple(
    "Relation",
    ["head", "head_type", "tail", "tail_type", "type"],
    defaults=["", "", "", "", ""])

NO_RELATION = "no relation"

relations = ['no_relation',
             'org:alternate_names',
             'org:city_of_headquarters',
             'org:country_of_headquarters',
             'org:dissolved',
             'org:founded',
             'org:founded_by',
             'org:member_of',
             'org:members',
             'org:number_of_employees/members',
             'org:parents',
             'org:political/religious_affiliation',
             'org:shareholders',
             'org:stateorprovince_of_headquarters',
             'org:subsidiaries',
             'org:top_members/employees',
             'org:website',
             'per:age',
             'per:alternate_names',
             'per:cause_of_death',
             'per:charges',
             'per:children',
             'per:cities_of_residence',
             'per:city_of_birth',
             'per:city_of_death',
             'per:countries_of_residence',
             'per:country_of_birth',
             'per:country_of_death',
             'per:date_of_birth',
             'per:date_of_death',
             'per:employee_of',
             'per:origin',
             'per:other_family',
             'per:parents',
             'per:religion',
             'per:schools_attended',
             'per:siblings',
             'per:spouse',
             'per:stateorprovince_of_birth',
             'per:stateorprovince_of_death',
             'per:stateorprovinces_of_residence',
             'per:title']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file',
                        help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(prediction)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        logger.info("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        logger.info("")

    # Print the aggregate score
    if verbose:
        logger.info("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    logger.info("Precision (micro): {:.3%}".format(prec_micro))
    logger.info("   Recall (micro): {:.3%}".format(recall_micro))
    logger.info("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def extract_predictions_for_scoring(
        pred_relations,
        gt_relations,
        entity_types,
        relation_types,
        mode,
        overlap_ratio,
        sort_predictions):
    rc_scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
    ner_scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in entity_types + ["ALL"]}

    no_gt_rels_found_ids = []

    # several predictions per sentence (or in the case of rcv2, per paragraph)
    for sent_idx, (pred_sent, gt_sent) in enumerate(zip(pred_relations, gt_relations)):
        # logger.info("****** Scoring sample text ******")

        gt_rel_types = [rel["type"] for rel in gt_sent]
        found_gt_rels_all_types = []
        last_gt_found_all_types = []

        for rel_type in relation_types:
            if mode != "overlap":
                if mode == "strict":  # strict mode takes argument types into account
                    pred_rels = [Rel(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent
                                 if rel["type"] == rel_type]
                    gt_rels = [Rel(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                               rel["type"] == rel_type]
                elif mode == "boundaries":  # boundaries mode only takes argument spans into account
                    pred_rels = [Rel(head=rel["head"], tail=rel["tail"]) for rel in pred_sent if
                                 rel["type"] == rel_type]
                    gt_rels = [Rel(head=rel["head"], tail=rel["tail"]) for rel in gt_sent if rel["type"] == rel_type]

            else:  # overlap mode takes only a % (overlap_ratio) of argument spans into account, rel_type still has to be correct
                # more precisely, overlap_ratio only applies to quote arguments, whose length might be difficult to match precisely
                # other arguments (agent, cues, etc.) are typically short and therefore we expect the model to extract them accurately

                pred_rels = [Rel(head=rel["head"], tail=rel["tail"], type=rel_type) for rel in pred_sent if
                             rel["type"] == rel_type]
                gt_rels = [Rel(head=rel["head"], tail=rel["tail"], type=rel_type) for rel in gt_sent if
                           rel["type"] == rel_type]

                # find predictions where the head matches (many matches are possible, e.g. "indique" rel heads (i.e. cues) are frequently the same)
                matching_head_rels_ids = [i for gt_rel in gt_rels for i, pred_rel in enumerate(pred_rels) if
                                          pred_rel.head == gt_rel.head]
                # if empty, ad since the type was good

                for i, gt_rel in enumerate(gt_rels):
                    if (rel_type == 'cité' or rel_type == 'indique'):
                        # look for a match of more than overlap_ratio*100% characters for the tail (quote)
                        for pred_rel in [pred_rels[i] for i in matching_head_rels_ids]:
                            match_tuple = SequenceMatcher(None, pred_rel.tail, gt_rel.tail).find_longest_match(0,
                                                                                                               len(pred_rel.tail),
                                                                                                               0,
                                                                                                               len(gt_rel.tail))
                            if match_tuple.size >= len(gt_rel.tail) * overlap_ratio and (
                                    len(pred_rel.tail) <= len(gt_rel.tail)):
                                gt_rels[i].tail = pred_rel.tail  # modify the gt tail so they match for a tp
                                break  # sanity check in case a quotes it's repeated (would be an annotation error)

            # Count TP, FP and FN per type
            pred_rels_set = set(pred_rels)
            gt_rels_set = set(gt_rels)

            # Get NER scores
            if mode == 'strict':
                print(pred_rels_set)
                pred_rels_set_etypes = {(p["head_type"], p["tail_type"]) for p in pred_rels_set}
                gt_rels_set_etypes = {(gt["head_type"], gt["tail_type"]) for gt in gt_rels_set}

                for ent_type in entity_types:
                    ner_scores[ent_type]["tp"] += len(pred_rels_set_etypes & gt_rels_set_etypes)  # set intersection
                    ner_scores[ent_type]["fp"] += len(pred_rels_set_etypes - gt_rels_set_etypes)  # set difference
                    ner_scores[ent_type]["fn"] += len(gt_rels_set_etypes - pred_rels_set_etypes)

            # Get RE scores
            rc_scores[rel_type]["tp"] += len(pred_rels_set & gt_rels_set)  # set intersection
            rc_scores[rel_type]["fp"] += len(pred_rels_set - gt_rels_set)  # set difference
            rc_scores[rel_type]["fn"] += len(gt_rels_set - pred_rels_set)

            # display some info about the predictions
            #logger.info(f"Info for relations of type {rel_type}")
            #if len(gt_rels) == 0:
            #     logger.info(f"Example cointained no relations of this type")
            #elif len(gt_rels) == len(pred_rels):
            #    logger.info(f"All {len(gt_rels)} relations of this type were predicted")
            #elif (len(gt_rels) == 0 and len(pred_rels) > 0) or (len(gt_rels) - len(pred_rels) < 0):
            #    logger.info(f"Exacly {abs(len(gt_rels) - len(pred_rels))} relations in this example were wrongly predicted with this type")
            #elif len(gt_rels) - len(pred_rels) > 0: # some might be incorrect predictions from other types
            #    logger.info(f"Missing {len(gt_rels) - len(pred_rels)}/{len(gt_relations)} relations for this type, maybe more")

            # we might also want to build a bool table with one row per example
            # wrong_head, wrong tail -> good to know for boundaries or strict
            # wrong_head_type, wrong_tail_type -> for strict

            # e.g. gt_rel_types = [indique, cité, réf, cite, cité] -> rel_type_ids = [1, 3, 4] (cités)
            # -> found_ids_with_type = [1, 3] -> last_gt_found_with_type = 3
            rel_type_ids = [i for i, item in enumerate(gt_rel_types) if item == rel_type]
            if rel_type_ids:
                found_ids_with_type = [rel_type_ids[i] for i, gt_rel in enumerate(gt_rels) if gt_rel in pred_rels]
                if found_ids_with_type:
                    found_gt_rels_all_types.append(len(found_ids_with_type))
                    last_gt_found_all_types.append(max(found_ids_with_type))
                    # logger.info(f"Index of last relation found for type {rel_type}: {max(found_ids_with_type)}")

        # logger.info(f"Index of the last relation *found* for this sample: {max(last_gt_found_all_types) if last_gt_found_all_types else 'no rel found'}"),
        # logger.info(f"Index of the last annotated relation: {len(gt_rel_types) - 1}")
        #logger.info(f"# correct: {sum(found_gt_rels_all_types)}, # predicted: {len(pred_sent)}, # gt: {len(gt_sent)}")
        if not last_gt_found_all_types:
            no_gt_rels_found_ids.append(sent_idx)

    logger.info(f"List of sample ids were no correct relation was predicted: {no_gt_rels_found_ids}")

    return rc_scores, ner_scores


def compute_pre_rec_f1(types, scores, mode, re=True, re_str_summary=None):
    # Compute per relation Precision / Recall / F1
    for type in scores.keys():
        if scores[type]["tp"]:
            scores[type]["p"] = 100 * scores[type]["tp"] / (scores[type]["fp"] + scores[type]["tp"])
            scores[type]["r"] = 100 * scores[type]["tp"] / (scores[type]["fn"] + scores[type]["tp"])
        else:
            scores[type]["p"], scores[type]["r"] = 0, 0

        if not scores[type]["p"] + scores[type]["r"] == 0:
            scores[type]["f1"] = 2 * scores[type]["p"] * scores[type]["r"] / (
                    scores[type]["p"] + scores[type]["r"])
        else:
            scores[type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[type]["tp"] for type in types])
    fp = sum([scores[type]["fp"] for type in types])
    fn = sum([scores[type]["fn"] for type in types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[type]["f1"] for type in types])
    scores["ALL"]["Macro_p"] = np.mean([scores[type]["p"] for type in types])
    scores["ALL"]["Macro_r"] = np.mean([scores[type]["r"] for type in types])

    if re:
        logger.info(f"RE Evaluation in *** {mode.upper()} *** mode")
        print(re_str_summary)
        print(tp)
        logger.info(re_str_summary + "correct: {}.".format(tp))
    else:
        logger.info(f"NER Evaluation in *** {mode.upper()} *** mode")
    logger.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    logger.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    logger.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for type in types:
        logger.info("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            type,
            scores[type]["tp"],
            scores[type]["fp"],
            scores[type]["fn"],
            scores[type]["p"],
            scores[type]["r"],
            scores[type]["f1"],
            scores[type]["tp"] +
            scores[type][
                "fp"]))

    return scores, precision, recall, f1


'''Adapted from: https://github.com/btaille/sincere/blob/6f5472c5aeaf7ef7765edf597ede48fdf1071712/code/utils/evaluation.py'''


def re_score(
        pred_relations,
        gt_relations,
        entity_types,
        relation_types,
        mode="boundaries",
        overlap_ratio=0.7,
        sort_predictions=True):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """
    assert mode in ["strict", "boundaries", "overlap"]
    relation_types = relations if relation_types is None else relation_types
    # relation_types = [v for v in relation_types if not v == "None"]

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    rel_type_counts = []

    for rel_type in relation_types:
        count = 0
        for item in gt_relations:
            for rel in item:
                if rel["type"] == rel_type:
                    count += 1
        rel_type_counts.append(count)
    logger.info(f"Evaluating on {len(gt_relations)} examples (multiple relations per example)")
    logger.info(" ".join(f"{count} of type {type}," for count, type in zip(rel_type_counts, relation_types)))

    # get TP, FP and FN counts per relation type
    rc_scores, ner_scores = extract_predictions_for_scoring(
        pred_relations,
        gt_relations,
        entity_types,
        relation_types,
        mode,
        overlap_ratio,
        sort_predictions)

    # compute P, R and F1
    # TODO: DEBUG
    #ner_scores, ner_prec, ner_recall, ner_f1 = compute_pre_rec_f1(entity_types, ner_scores, mode,
    #                                                              re_str_summary=f"processed {n_sents} sentences with {n_rels} relations; found: {n_found} relations; ")
    rc_scores, rc_prec, rc_recall, rc_f1 = compute_pre_rec_f1(relation_types, rc_scores, mode,
                                                              re_str_summary=f"processed {n_sents} sentences with {n_rels} relations; found: {n_found} relations; ")

    return rc_scores, rc_prec, rc_recall, rc_f1


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        logger.info("Gold and prediction file must have same number of elements: {} in gold vs {} in prediction".format(
            len(key), len(prediction)))
        exit(1)

    # Score the predictions
    score(key, prediction, verbose=True)

