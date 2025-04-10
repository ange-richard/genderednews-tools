# Computing Mentions Masculinity

## Setting up your environment

All commands are run from the `mentions_masc_computing/` directory.

```shell
  conda create -n "gn_mentionsmasc" python=3.9.12 pip=20.3.1
  conda activate gn_mentionsmasc
  pip install -r requirements.txt
```

## Usage

Expected input file:
- `csv` (use `--input-sep` argument to specify separator, default is ;) or a `jsonl` file;
- One text per line
- At least one "text" column (use `--text-field` argument to specify another text fieldname)

Example command:
```shell
python compute_mentions_masc.py test.csv --input-sep "," --write-to-file
```

Output file:
- `csv` file with ";" as a separator
- Will contain original columns as well as two additional ones: `mentions_masc` (the mentions masculinity of names found) and `genderized_names` (names that were extracted and given a masculinity score)

## References

We use this metric in several of our works, namely in [Courson _et al_ (2024)](https://osf.io/preprints/socarxiv/j7ydu_v1) and on the _GenderedNews_ dashboard (link TBA).
