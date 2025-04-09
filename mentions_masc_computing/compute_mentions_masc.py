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
    if len(doc) <= 500:
        sample = doc[:500]
        vocab_richness = len(set([tok.text.lower for tok in sample]))/len(sample)
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
        return m_rate, _names, vocab_richness
    return m_rate, _names, vocab_richness


def add_mentions_masc(input_f, output_f, nlp, text_field, write_to_file=False):
    df = pd.read_csv(input_f, sep=',')
    print(f"{datetime.datetime.now()}: Started mentions masc computing of {input_f}...")
    df[["mentions_masc", "genderized_names","vocab_richness"]] = df.apply(lambda x: compute_mentions_masc(x[text_field], nlp),
                                                         axis=1, result_type="expand")

    if output_f == "default":
        output_f = f"{input_f}_mentions_masc.csv"
    if write_to_file:
        print(f"{datetime.datetime.now()}: Saving to file ...")
        df.to_csv(output_f, sep=";", index=False)
    # return df


def get_args():
    description = """Takes a csv with a \"text\" field and computes the mentions masculinity for each text. \n
                    WARNING: This script needs to be placed in the same directory as a databases/ folder containing the 
                    insee_sexratio_firstnames_2017.csv file to run properly.\n
                    Is also needed at least the fr_core_news_md spacy model"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", type=str,
                        help="Input file with each line an article (has to be a csv with each line an article entry, with at least one \"text\" field")
    parser.add_argument("--output-file", type=str,
                        help="The name of the output file (will be identical as the input file, with an additional \"mentions_masc\" column",
                        default="default")
    parser.add_argument("--spacy-model", type=str,
                        default="fr_core_news_md")
    parser.add_argument("--write-to-file", action="store_true")
    parser.add_argument("--text_field", type=str, default="text")
    return parser.parse_args()


def main():
    args = get_args()
    print("Loading nlp model ...")
    nlp = spacy.load(args.spacy_model)
    add_mentions_masc(args.input_file, args.output_file, nlp, args.text_field, args.write_to_file)


if __name__ == "__main__":
    main()
