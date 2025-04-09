#!/usr/bin/env python
# coding: utf-8
"""
Shared script to genderize input quotes. Depending on the input format (GenderGapFrench, Dygie++ or REBEL), run the
corresponding script "genderize_{architecture name}
"""
import csv
import re

import requests.exceptions
import spacy
import pandas as pd
import statistics

#First names libraries
from genderize import Genderize
from genderize import GenderizeException
import gender_guesser.detector as gender
d = gender.Detector()

# CHANGE WITH ABSOLUTE PATHS
names_df = pd.read_csv("utils/insee_sexratio_firstnames_2017.csv", delimiter=";")
with open("utils/genderize_cache.csv", "r") as cache:
    genderize_reader = csv.DictReader(cache)
    genderize_dict = list(genderize_reader)
genderize_cache = dict(zip(map(lambda x: x["name"], genderize_dict), map(lambda x: (x["gender"], x["probability"]), genderize_dict)))

#jobs dictionary
jobs_df = pd.read_csv("utils/occupations_clean.csv", delimiter=";",index_col=0, header=None)
jobs_dict=jobs_df.to_dict()
nlp = spacy.load("fr_core_news_md")


# utils
def check_genderize_cache(name):
    if name in genderize_cache.keys():
        return genderize_cache[name]
    else:
        try:
            gen = Genderize().get([name])
            with open("utils/genderize_cache.csv",
                      "a") as add_name_to_cache:
                add_name_to_cache.write(f"{name},{gen[0]['gender']},{gen[0]['probability']}\n")
            return (gen[0]["gender"],gen[0]["probability"])
        except GenderizeException:
           return [None]
        except requests.exceptions.ConnectTimeout:
            return [None]


def remove_titles(txt):
    """Method to clean special titles that appear as prefixes or suffixes to
       people's names (common especially in articles from British/European sources).
       The words that are marked as titles are chosen such that they can never appear
       in any form as a person's name (e.g., "Mr", "MBE" or "Headteacher").
    """
    honorifics = ["Mr", "Ms", "Mrs", "Miss", "Dr", "Sir", "Dame", "Hon", "Professor",
                  "Prof", "Rev", "Me", "M", "M.","Monsieur", "Madame","Mme", "Mlle", "Mademoiselle",
                  "Messieurs", "MM.", "MM", "Mgr"]
    titles = ["Père", "Frère", "S(œ|oe)ur","Mère","Grand-père", "Grand-mère"]
    extras = ["et al", "www", "href", "http", "https", "Ref", "rel", "eu", "span"]
    banned_words = r'|'.join(honorifics + titles + extras)
    pattern = f'\\b({banned_words})\\b'
    txt = txt.text
    if txt.isupper():
        txt = txt.lower().capitalize()
    txt = re.sub(pattern,'', txt)
    return txt.strip()


# checker functions
def check_title(speaker):
    speaker = speaker.lower()
    title_gender = None
    honorifics_m = r"|".join(["mr", "m(\.)?", "monsieur", "sir", "père", "frère","mgr", "monseigneur", "prince", "général", "lord", "comte"])
    honorifics_f = r"|".join(["ms", "mrs", "miss", "madame", "mme", "mlle", "mademoiselle", "m(e|è)re", "s(œ|oe)ur", "générale", "lady", "princesse", "comtesse"])
    if re.search(re.compile(f"\\b({honorifics_m})\\b"), speaker) is not None:
        title_gender = 1
    elif re.search(re.compile(f"\\b({honorifics_f})\\b"), speaker) is not None:
        title_gender = 0
    return title_gender


def check_ner(speaker):
    speaker = re.sub('[&\"\/\(\)=+\}\{*\.#^$£!:;?,§~\[\]`<>]', ' ', speaker).strip()
    doc = nlp(speaker)
    ents = [ent for ent in doc.ents if ent[0].ent_type_ == "PER"]
    if len(ents) > 1:
        # if there are several people -> Group_of People. For now, considering it as unknown but:
        # TODO: genderize groups of people
        return None
    elif len(ents) == 1:
        candidate_speaker = ents[0]
        speaker = remove_titles(candidate_speaker)
        try:
            prenom = speaker.split()[0]
            #Genderize method
            g = check_genderize_cache(prenom)
            if g[0] is not None:
                gen = g[0]
                if gen == "male" and float(g[1]) >= 0.7:
                    return 1
                elif gen == "female" and float(g[1]) >= 0.7:
                    return 0
                elif float(g[1]) < 0.7:
                    return 0.5
            else:
                try:
                    gender_score_name = names_df.loc[(names_df["preusuel"]==prenom.lower()),"sexratio_prenom"].values[0]
                    return float(gender_score_name.replace(",","."))
                except (KeyError, IndexError):
                    # Else: gender_guesser method
                    ner_gender = None
                    gender = d.get_gender(prenom)
                    if gender == "female":
                        ner_gender = 0
                    elif gender=="mostly_female":
                        ner_gender = 0.2
                    elif gender in ["andy"]:
                        ner_gender = 0.5
                    elif gender == "mostly_male":
                        ner_gender = 0.8
                    elif gender == "male":
                        ner_gender = 1
                    return ner_gender
        except IndexError:
            return None
    else:
        return None


def check_job(speaker):
    """
    Checks if there's a genderized job through 2 ways
        - gendered det followed by neutral function
        - gendered job/occupation name
    :param speaker:
    :return:
    """
    job_gender = None
    det_jobs_neutral = ["secrétaire", "porte-parole", "ministre", "député","responsable","diplomate","commissaire","analyste"]
    det_jobs_m = r"|".join(
        ["correspondant","chef"]+det_jobs_neutral)
    det_jobs_f = r"|".join(["correspondante","cheffe"]+det_jobs_neutral)
    if re.search(re.compile(f"\\b(un|le|ce(t?))\\b ({det_jobs_m})"), speaker.lower()) is not None:
        return 1
    elif re.search(re.compile(f"\\b(une|la|cette)\\b ({det_jobs_f})"), speaker.lower()) is not None:
        return 0
    else:
        nlped_speaker = nlp(speaker)
        # TODO: check if without NLP that would work better ?
        nouns = [noun.lower_ for noun in nlped_speaker if noun.pos_ == "NOUN"]
        for noun in nouns:
            if noun in jobs_dict.keys():
                job_gender = jobs_dict[noun]
        return job_gender


def genderize_speaker(speaker, speaker_label=None, speaker_gender=None, speaker_antec=None, speaker_antec_label=None):
    if speaker_gender:
        return speaker_gender
    # Unknown cases
    elif (not speaker) | (speaker == ""):
        return None
    elif speaker_label in ["Source_pronoun", "<pron>"] and speaker_antec_label in ["Organization", "<org>"]:
        return None
    elif speaker_label in ["Group_of_People", "<peop>"]:
        # TODO: Group_of_People -> try to genderize ?
        return "Other"
    elif not speaker_label and speaker.lower() in ["il","ils","elle","elles","on"]:
        return "Unknown"
    # Agent speaker cases
    else:
        if speaker_label in ["Source_Pronoun", "<pron>"] and speaker_antec is not None:
            speaker_text = speaker_antec
        else:
            speaker_text = speaker
        indices_list = {
            "title": check_title(speaker_text),
            "ner": check_ner(speaker_text),
            "jobs": check_job(speaker_text)
        }
        try:
            m = statistics.mean([ind for ind in indices_list.values() if ind is not None])
            if m > 0.7:
                return "Male"
            elif m < 0.3:
                return "Female"
            else:
                return "Unknown"
        except statistics.StatisticsError:
            return "Unknown"
