""" This script can be used to take as input a data file (csv or jsonl)
and output a .jsonl file formatted for prediction by REBEL pipeline
TODO: you need to change lines 11 to 18 to read your own file in a pandas dataframe
"""
import argparse
import pandas as pd


def main(input_file, output_file):
    #################### CHANGE BELOW TO MATCH YOUR DATA FORMAT ##############################
    if input_file.endswith(".jsonl"):
        data_df = pd.read_json(input_file, orient="records", encoding="utf-8", lines=True)
    elif input_file.endswith(".csv"):
        data_df = pd.read_csv(input_file, encoding="utf-8", sep=";")
    else:
        raise ValueError(f"Unsupported file type: {input_file}")
    # if needed: rename columns (we need an 'id' and a 'text' column
    data_df.rename(columns={"youridfield": "id", "yourtextfield": "text"}, inplace=True)
    #################### CHANGE ABOVE TO MATCH YOUR DATA FORMAT ##############################
    data_rebel_df = data_df[["id", "text"]]
    # adding a newline at the end is needed for cutting into 512 tokens later.
    data_rebel_df['text'] = data_rebel_df['text'].apply(lambda x: x + "\n")
    # to jsonl
    data_rebel_df.to_json(output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to rebel format")
    parser.add_argument("dataset", help="Dataset file")
    parser.add_argument("output_file", help="Rebel formatted output file",
                        default="dataset_rebel.jsonl")
    args = parser.parse_args()
    main(args.dataset, args.output_file)
