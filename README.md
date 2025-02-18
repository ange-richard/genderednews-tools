# genderednews-tools
Example command for `compute_mentions_masc.py`:

```python compute_mentions_masc.py test.csv --output-file output.csv --write-to-file```

Input file has to be a csv file (set separator is ",", see line 47 to modify separator), one line per text, with at least a text column. Default name for the field is "text", can be changed with the `--text_field` command.

Output file will be the same csv file with ";" separator with three additional columns: 
  -`mentions_masc` (the mentions masculinity of found names), 
  -`genderized_names` (names that were extracted and given a masculinity score based on the INSEE database),
  -`vocab_richness` (an indicator of the vocabularity diversity).
