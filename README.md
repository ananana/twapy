# twapy: Temporal Word Analogies in Python
# this fork: twapy adapted for word analogies for comparing language varieties

This package contains Python code to build and evaluate models for studying word analogies: pairs of words from different language varieties that are used with similar meanings.

## Requirements

* gensim
* sklearn
* nltk + the WordNet corpus

## Usage

To produce a csv containing word analogies, run `find_best_analogies.py`
Arguments: binary files containing embeddings for he 2 corpora respectively.


