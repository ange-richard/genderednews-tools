# Computing Quotes Masculinity
This repository is a simplified and minimal version of the [REBEL repository](https://github.com/Babelscape/rebel/tree/main), which is the framework we adapted to train our own quote extraction model for French. Please refer to [our article](https://aclanthology.org/2024.lrec-main.654/) published at LREC-COLING 2024 for details on the corpus used.

## Setting up your environment

Follow each step of the setup/running of the code with care.
All commands are run from the `citations_masc_computing/` directory.

### Create conda environment

Be aware that the REBEL framework needs a specific `python` version (<3.11) and a specific `pip` version (<24). Using latest versions of both libraries will result in conflicts with the necessary `pytorch-lightning` version. The following command contains the versions we use to run the code.

```shell
  conda create -n "gn_citmasc" python=3.9.12 pip=20.3.1
  conda activate gn_citmasc
```

### Install required libraries

```shell
  pip install -r requirements.txt
```
Double-check that the `pytorch-lightning` installation has not overriden any other installed libraries such as `torch`. If necessary, install back the right versions individually as such:
```shell
    pip install torch==1.11.0
```

### Prepare the repository

1. Download our citations extraction model on [Zenodo](https://doi.org/10.5281/zenodo.15189896) and place it in `src/checkpoint/`
2. Place your data in `data/`
3. Change inference file, output file and checkpoint path variables with the corresponding absolute paths in [conf/root_infer.yaml](conf/root_infer.yaml)

## Usage

Below is the detailed usage of the pipeline of quotation extraction and genderization.

### Preprocess your data

1. Modify the preprocessing script for your own data

    The [preprocessing/data_to_rebel.py](preprocessing/data_to_rebel.py) is a template to transform your data into the desired REBEL input format. Please modify the script to read your input data in the dedicated spot in the script. Minimal required fields are `id` and `text`.
2. Format your data into the desired input for the model
```shell
    cd preprocessing/
    python data_to_rebel.py ../data/[yourdata.jsonl] ../data/[yourdata_REBELformat].jsonl
```
3. Split the data entries so they all fit into the allowed 512-tokens length input
```shell
cd src/utils/
python dataset_utils.py --input-dir ../data/[yourdata_REBELformat].jsonl --output-dir ../data/ [--output-file yourdata_REBELformat_512cuts.jsonl]
```
This script splits the entries longer than 512 tokens into several entries by cutting at the newline or punctuation mark closest to the 512 tokens mark. Default output file name is `[yourdata_REBELformat]_extended_dataset.jsonl`.

### Run prediction

```shell
    python src/predict.py
```

Default is `cpu` use. You can change to `gpu` by passing it to the `accelerator` argument of the `Trainer`.

### Post process your data

Postprocessing step will allow you to:
- Reaggregate the entries that were split into 512-tokens long chunks at the preprocessing step
- Match back the predicted output to input ids
- Optional: Genderize the quote speakers

```shell
cd postprocesing/
python genderize_and_aggregate.py ../data/[prediction_output_filename].jsonl ../data/[yourdata_REBELformat]_extended_dataset.jsonl [--add-gender]
```
Please note that the two positional arguments are the predicted output filename and the file used as an input for prediction (your data after split on 512 tokens)
This will input two files:
- **[prediction_output_filename]_agg.jsonl**: Each entry is an input text. ids are added back to re-aggregated entries, and gender is added as a key to each quote if option add-gender was used
- **[prediction_output_filename]_quote-list.csv**: Each line is a quote (matched with text id)


