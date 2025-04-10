# GenderedNews Tools

This repository contains the code to the tools used to compute two metrics used in the _GenderedNews_ project that looks at gender inequality in French newstexts. Please refer to the latest version of [this article](https://arxiv.org/abs/2202.05682) for details on how these metrics are computed. This repository is divided in two parts, one for each tool:
- Mentions masculinity computing 
- Citation masculinity computing

## Mentions masculinity

The mentions masculinity computes what we call the `masculinity rate` of first names mentioned in (a) given newstext(s). The `mentions_masc_computing` contains the code that identifies first names (based on `Named Entity Recognition`) in a text and assigns a masculinity rate (derived from the INSEE first names database). 

## Citations masculinity

The citation masculinity computes the proportion of men quoted in (a) given newstext(s). The `citation_masc_computing` contains the code to extract quotes then genderize the extracted speakers. It is based on an adaptation of the [REBEL](https://github.com/Babelscape/rebel/tree/main) framework on French Quotation Extraction (see [our article](https://aclanthology.org/2024.lrec-main.654/) published at LREC-COLING for more details).

## Setting up environments

For simplicity, we advise to create a separate environment for each tool. See respective `README.md`s and `requirements.txt`s for instruction on how to set up and run each code.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE.md)
License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

First versions of these tools were implemented by [Gilles Bastin](https://github.com/gillesbastin) (for mentions masculinity) and by Laura Alonzo (for citation masculinity).