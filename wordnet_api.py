from nltk.corpus import wordnet
import re

LEMMAS = {"EN" : list(wordnet.all_lemma_names(lang="eng")),
            "ES": list(wordnet.all_lemma_names(lang="spa")),
            "CA": list(wordnet.all_lemma_names(lang="cat"))}

# association between language codes as defined in this project, and wordnet
LANGUAGES = {"EN" : "eng",
            "FR" : "fra",
            "IT" : "ita",
            "ES" : "spa",
            "CA" : "cat"}

def all_words(language):
    '''Returns list of all words in a language
    :param language: language code
    '''

    all_words_wordnet = list(wordnet.all_lemma_names(lang=LANGUAGES[language]))
    all_words = map(lambda w: normalize_word(w), all_words_wordnet)
    return all_words

def is_word(word, language):
    '''Checks if word form is a valid word in the language.
    Strictly, that is the exact word form must be a valid entry in the dictionary.
    :param word: word form to look for (string)
    :param language: language word is in. must be one of the keys in LEMMAS
                    is meant to be coherent with language codes in models.py
    :return: True or False
    '''

    # check if it's in lemma list
    return normalize_wordnet(word) in LEMMAS[language]

def is_word_weak(word, language):
    '''Checks if word is a valid word in the language.
    Weakly, that is it may be a derivationally related form of an
    actual entry in the dictionary.
    :param word: word form to look for (string)
    :param language: language word is in. must be one of the keys in LEMMAS
                    is meant to be coherent with language codes in models.py
    :return: True or False
    '''

    # check if there is a synset associated with this word form
    return not not wordnet.synsets(normalize_wordnet(word), lang=LANGUAGES[language])

def normalize_word(word):
    '''Normalize word form to be compatible with word format
    in the project's DB
    :param word: original word form as found in wordnet
    :return: normalized word form
    '''

    # replace "_" with spaces
    return re.sub("_", " ", word)

def get_canonical_form(word):
    '''Get canonical form of this word,
    according to wordnet's derivationally related forms
    :return: string represeting canonical form of word if found;
            None otherwise
    Only works for english
    '''
    try:
        form = wordnet.morphy(word)
    except UnicodeDecodeError:
        form = wordnet.morphy(word.decode('utf-8'))
    if not form:
        return None
    return normalize_word(form)

def normalize_wordnet(word):
    '''Normalize word form to be compatible with wordnet
    (not as far as derivationally related forms. just the format)
    :param word: original word form as found in wordnet
    :return: normalized word form as found in wordnet
    '''

    # replace spaces with "_"
    return re.sub("\s+", "_", word)

def get_synonyms(word_in, language):
    '''Get synonyms list for this word, according to the requirements
    of this project - that is, take into account derivationally related forms
    only for words that are not valid lemmas
    :param word: original word
    :param language: language code
    :return: list of lists of synonyms for word (normalized); one list for each sense
    '''

    word = normalize_wordnet(word_in)

    lemmas = {}

    # if it has synsets for this exact word form, only get those (not those of derivationally related forms)
    if is_word(word, language):
        ss = wordnet.synsets(word, lang=LANGUAGES[language])
        # eliminate synsets with no lemmas for this language
        ss = filter(lambda s: s.lemma_names(lang=LANGUAGES[language]), ss)
        # get lemma names
        ss = filter(lambda s: word in s.lemma_names(lang=LANGUAGES[language]), ss)
        # eliminate word itself from list of synonyms, flatten synonyms list
        for s in ss:
            lemmas[s.name()] = \
                {'POS' : s.pos(),
                'synonyms' : [normalize_word(lemma) for lemma in s.lemma_names(lang=LANGUAGES[language]) if lemma!=word]
                }
        return {s: d for (s, d) in lemmas.items() if d['synonyms']}
    # if there are no synsets for this exact word form, try derivationally related forms
    # only for english
    else:
        if language != "EN":
            return {}
        ss = wordnet.synsets(word, lang=LANGUAGES[language])
        # eliminate word itself from list of synonyms, flatten synonyms list
        for s in ss:
            lemmas[s.name()] = \
                {'POS' : s.pos(),
                'synonyms' : [normalize_word(lemma) for lemma in s.lemma_names(lang=LANGUAGES[language]) if lemma!=wordnet.morphy(word)]
                }
        return {s: d for (s, d) in lemmas.items() if d['synonyms']}
        # return filter(lambda l: l, lemmas)
    
def get_synonyms_list(word, language="EN"):
    synonyms_dict = get_synonyms(word, language)
    synonyms_list = list(set(sum([synonyms_dict[synset]['synonyms'] for synset in synonyms_dict], [])))
    return synonyms_list