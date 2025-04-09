"""
Postprocessing of written output of predict.py
This script:
- Adds back the unique ID to each input text.
- Merges back split texts together (in case they were cut at preprocessing stage)

- Transforms the "prediction field" into a "quotes" field (type: List).
    Each quote is an object with "source","content","cue", "source_type" and "quote_type" fields
- Optional but highly recommended: --add-gender option performers genderization of quotes
"""
import argparse
import json
from genderize_quotes import genderize_speaker
import pandas as pd


############### Quote CLASS ###################
class Quote(object):
    __slots__ = ("content", "quote_type", "quote_length", "cue", "source", "source_type", "gender")

    def __init__(self, cite, indiques, add_gender):
        self.content, self.quote_type = cite["tail"], cite["tail_type"]
        self.quote_length = len(self.content)
        self.cue = self.get_cue(indiques, cite)
        self.source, self.source_type = cite["head"], cite["head_type"]
        self.gender = genderize_speaker(self.source, self.source_type) if add_gender else None

        # Assertions should be uncommented and used to manually correct data
        """assert self.quote_type in ["<dirquot>", "<indquot>", "<mixquot>"], \
            f"{self.content} type not a quote but type {self.quote_type}"
        assert self.source_type in ["<pron>", "<per>", "<org>", "<peop>"], \
            f"{self.source} type not a source but type {self.source_type}"
            """

    @staticmethod
    def get_cue(candidate_indiques, cite_rel):
        # TODO: We filter by head type to only get "cue" ents but maybe remove "cue" filter ?
        corr_indique = [rel for rel in candidate_indiques if rel["tail"] == cite_rel["tail"] and rel["head_type"] == "<cue>"]
        # removing duplicates (?)
        unique_corr_indique = list({v['head']: v for v in corr_indique}.values())
        if unique_corr_indique:
            # I honestly don't remember why I wrote it like this but there must be a reason I guess.
            indique = unique_corr_indique[0]["head"] if "head" in unique_corr_indique[0] else None
        else:
            indique = None
        # Assertions should be uncommented to manually correct data.
        """
        if indique:
            assert len(unique_corr_indique) == 1, \
            f"More than one possible 'indique' corresponding for quote \'{cite_rel}\':\n {unique_corr_indique}"
        """
        return indique

    def q_to_dict(self):
        return {
            "content": self.content,
            "quote_type": self.quote_type,
            "quote_length": self.quote_length,
            "cue": self.cue,
            "source": self.source,
            "source_type": self.source_type,
            "gender": self.gender
        }


################ HELPER FUNCS #################################
def merge_by_quote(predictions, add_gender):
    """
    Merge predictions into a single quote object.
    :param predictions: list of predictions for an input
    :return: List of Quote object associated with input
    """
    cite = [rel for rel in predictions if rel["type"] == "cité"
            and "cité <indquot>./bds" not in rel["head"]  # corpus-specific filter, can be removed
            ]
    indiques = [rel for rel in predictions if rel["type"] == "indique"]
    # join by tail
    quotes = []
    for c in cite:
        # TODO: Probably not the fastest way to create Object + slots then transform to dict. Test create dict directly ?
        quotes.append(Quote(c, indiques, add_gender).q_to_dict())
    return quotes


def get_citmasc(w,m):
    try:
        return m / w if (m+w) > 0 else None
    except ZeroDivisionError:
        return 1
    except (KeyError, TypeError):
        return None


def compute_quote_stats(quotes):
    nb_women = len([q for q in quotes if q["gender"] == "Female"])
    nb_men = len([q for q in quotes if q["gender"] == "Male"])
    nb_quotes = nb_men + nb_women
    citmasc = get_citmasc(nb_women, nb_men)
    return nb_quotes, nb_women, nb_men, citmasc


################ READING FUNC ###############################
def read_jsonl(file):
    with open(file, encoding="utf8") as f:
        entries = [json.loads(line) for line in f]
    return entries


##################### MAIN ###############################
def main(pred_file, og_file, add_gender):

    # 1: Add back ids to each prediction
    preds = read_jsonl(pred_file)
    og = read_jsonl(og_file)
    # We know the model predicts sequentially so we merge on rank
    dataset = [{"id":input_entry["id"],
                "quotes":pred_entry["predictions"]} for input_entry, pred_entry in zip(og,preds)]

    # 2: Merge relations into quote objects
    # (to a list of quotes instead of a list of relations) and genderize
    dataset_df = pd.DataFrame(dataset, columns=["id","quotes"])
    dataset_df["quotes"] = dataset_df["quotes"].apply(lambda x: merge_by_quote(x, add_gender))

    # 3: Re-aggregate splitted texts
    # Here you can filter out quotes as you wish with if condition (ex: only gendered quotes, only <org> quotes, etc)
    dataset_df.rename(columns={"id": "uno"}, inplace=True)
    deextended_df = dataset_df.groupby("uno").agg(
        {"quotes": lambda x: [q for u in list(x)
                              for q in u # if ((q["gender"] in ["Male", "Female"]) | (q["source_type"] == "<org>"))
                              ]}).reset_index()

    # 4: Calculate additional stats about gender
    if add_gender:
        deextended_df[["nb_quotes", "nb_women", "nb_men", "citmasc"]] = deextended_df.apply(lambda x: compute_quote_stats(x["quotes"]),axis=1, result_type="expand")

    # 5: Write output
    output_ext = pred_file.rsplit(".",1)[0]

    # Jsonl file with one entry a document (fields "uno" and "quotes")
    ext = "gender" if add_gender else "agg"
    deextended_df.to_json(f"{output_ext}_{ext}.jsonl", orient="records", lines=True)

    # Csv file with one line per quote
    quotes_df = pd.json_normalize(deextended_df.to_dict(orient="records"), "quotes", ["uno"])
    quotes_df = quotes_df.drop_duplicates()
    quotes_df.to_csv(f"{output_ext}_quote-list.csv", index=False, sep=";")

    print(f"Wrote {output_ext} stats per article and quote list.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", type=str,
                        help="The jsonl file containing predictions (output of predict.py)")
    parser.add_argument("origin_file", type=str,
                        help="The jsonl file containing texts before prediction")
    parser.add_argument("--add-gender", action="store_true", default=False)
    args = parser.parse_args()
    main(args.prediction_file, args.origin_file, args.add_gender)