import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import csv
import re
import string

class LemmaTokenizer(object):
    def __init__(self, lang):
        self.stemmer = SnowballStemmer(lang)
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) 
            if not all([c in string.punctuation for c in t]) and not re.match('[0-9]+', t)]

class SimpleTokenizer(object):
    '''removes punctuation and numbers'''
    def __init__(self):
        pass
    def __call__(self, doc):
        # TODO: why is doc unicode??
        # if isinstance(doc, unicode):
        #     doc = doc.encode("utf-8")
        return [t for t in self._tokenize(doc) 
            if not all([c in string.punctuation for c in t]) and not re.match('[0-9]+', t)]
    def _tokenize(self, doc):
        # print filter(None, re.split(re.compile(r'[\W\d]', re.UNICODE), doc))
        return filter(None, re.split(re.compile(r'[\W\d]', re.UNICODE), doc))


def get_word_classifier(stopwords = None, vocabulary = None, stem = False, clean=True, max_features=None, kbest=None):
    ''':param stopwords: list of stopwords
    :param stem: stem words on tokenization
    :param clean: remove punctuation and numbers
    :param max_features: only use most frequent max_features words
    '''
    # clf = MultinomialNB()
    clf = SVC(kernel='linear')#, class_weight="auto")
    # clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    if stem:
        cv = CountVectorizer(stop_words=stopwords, vocabulary=vocabulary, tokenizer=LemmaTokenizer(), max_features=max_features)
    elif clean:
        cv = CountVectorizer(stop_words=stopwords, vocabulary=vocabulary, tokenizer=SimpleTokenizer(), max_features=max_features)
    else:
        cv = CountVectorizer(stop_words=stopwords, vocabulary=vocabulary, max_features=max_features)
    # TODO: tune k
    if kbest:
        anova_filter = SelectKBest(k=kbest)
        text_clf = Pipeline([('vect', cv),
                              ('tfidf', TfidfTransformer()),
                              ('kbest', anova_filter),
                              ('clf', clf),
                              ])
    else:
        text_clf = Pipeline([('vect', cv),
                              ('tfidf', TfidfTransformer()),
                              ('clf', clf),
                              ])

    return text_clf


def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
 
    classifier.fit(X_train, y_train)
    print "Accuracy: %s" % classifier.score(X_test, y_test)
    return classifier

def load_texts(filepath):
    with open(filepath) as f:
        texts = [line.strip() for line in f]
    return texts

def load_vocabulary(dirpath=".", lang1="en", lang2="de"):
  vocabulary = []
  filepath = dirpath + "/" + "top_similarities_europarl_%s_%s.csv" % (lang1, lang2)
  with open(filepath, 'rb') as f:
      reader = csv.reader(f)
      headers = reader.next()
      for row in reader:
        vocabulary.append(row[0])
        vocabulary.append(row[1])
  return set(vocabulary)

def build_training_set(lang1="en", lang2="de", dir_path="models"):
    filename1 = dir_path + "/" + "europarl.%s.en.txt" % lang1
    filename2 = dir_path + "/" + "europarl.%s.en.txt" % lang2
    texts1 = load_texts(filename1)
    texts2 = load_texts(filename2)
    labels1 = [lang1 for l in texts1]
    labels2 = [lang2 for l in texts2]
    return texts1 + texts2, labels1 + labels2

if __name__ == '__main__':
    vocabulary = load_vocabulary()
    X, y = build_training_set()

    print "Classifier with vocabulary:"
    clf = get_word_classifier(vocabulary=vocabulary)
    train(clf, X, y)

    print "Classifier with all words"
    clf = get_word_classifier(vocabulary=vocabulary)
    train(clf, X, y)