import argparse
import datetime
import pandas as pd
import spacy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(".."))))

def get_names_df():
    name_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'databases/insee_sexratio_firstnames_2017.csv')
    names_df = pd.read_csv(name_file, sep=';')
    names_df = names_df.rename(columns={'preusuel': 'word'})
    names_df["sexratio_prenom"] = names_df["sexratio_prenom"].apply(lambda x: float(x.replace(",",".")))
    return names_df


def compute_mentions_masc(text, nlp=spacy.load('fr_core_news_md')):
    names_df = get_names_df()
    m_rate = None
    _names = None
    vocab_richness = None
    text = str(text)
    doc = nlp(text)
    try:
        ents = [ent.text for ent in doc.ents if ent[0].ent_type_ == "PER"]
        if ents:
            # This assumes that the first ent is always the first name, which is not always the case
            flattened_ents = [ent.split()[0].strip("«».").lower() for ent in ents
                              if ent.split()[0].strip("«».").lower() not in ["m", "mme"]]
            if flattened_ents:
                ner_tokens = pd.DataFrame(flattened_ents, columns=["word"])
                txt_tokens_with_name = pd.merge(ner_tokens, names_df, how='left').dropna(
                    subset=["sexratio_prenom"], inplace=False)
                _names = ",".join(txt_tokens_with_name['word'].tolist())
                m_rate = txt_tokens_with_name['sexratio_prenom'].mean()
    except TypeError:
        return m_rate, _names
    return m_rate, _names


def read_input(filename, sep):
    if os.path.isfile(filename) and filename[-3:] in ["csv","tsv"]:
        with open(filename, "r") as f:
            data = pd.read_csv(f, encoding="utf-8", sep=sep,
                               # If comma is used as a separator, specify columns names
                               # names=[],
                               header=0,
                                skip_blank_lines=True)
    elif os.path.isfile(filename) and filename.endswith("jsonl"):
        with open(filename, 'r') as f:
            data = pd.read_json(f, lines=True, orient="records", encoding="utf-8")
    else:
        sys.exit("The format of your data file is not recognized. It should be either csv, tsv or jsonl.")
    return data

def get_args():
    description = """Takes a csv with a \"text\" field and computes the mentions masculinity for each text. \n
                    WARNING: This script needs to be placed in the same directory as a databases/ folder containing the 
                    insee_sexratio_firstnames_2017.csv file to run properly.\n
                    Is also needed at least the fr_core_news_md spacy model"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", type=str,
                        help="Input file with each line an article (has to be a csv with each line an article entry, with at least one \"text\" field")
    parser.add_argument("--input-sep", type=str, help="If a csv file is used as input, specify the separator here. Avoid comma. Default is ;.", default=";")
    parser.add_argument("--output-file", type=str,
                        help="The name of the output file (will be identical as the input file, with an additional \"mentions_masc\" column",
                        default="default")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--write-to-file", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    print("Loading nlp model ...")
    nlp = spacy.load("fr_core_news_md")
    input_f = args.input_file

    df = read_input(input_f, args.input_sep)
    print(f"{datetime.datetime.now()}: Started mentions masc computing of {input_f}...")
    df[["mentions_masc", "genderized_names"]] = df.apply(
        lambda x: compute_mentions_masc(x[args.text_field], nlp),
        axis=1, result_type="expand")

    if args.output_file == "default":
        output_f = f"{input_f.split('.')[0]}_mentions_masc.csv"
    if args.write_to_file:
        print(f"{datetime.datetime.now()}: Saving to file {output_fsplit} ...")
        df.to_csv(output_f, sep=";", index=False)


if __name__ == "__main__":
    main()
