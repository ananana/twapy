import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection.univariate_selection import SelectKBest
import csv
import re
import string
import numpy as np
import scipy

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


def get_word_classifier(stopwords = None, vocabulary = None, stem = False, clean=False, max_features=None, kbest=None):
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
    return load_text_chunks(filepath)

def load_text_chunks(filepath, words_in_chunk=2000):
    with open(filepath) as f:
        sentences = [line.strip() for line in f]
    print "reading files..."
    words = []
    for s in sentences:
        words.extend(s.split())
    print "building text..."
    # build chunks of 2000 words texts
    texts = [" ".join(words[i:i+words_in_chunk]) for i in range(0, len(words), words_in_chunk)]
    return texts

def load_sentences(filepath):
    with open(filepath) as f:
        texts = [line.strip() for line in f]
    return texts

def load_vocabulary(languages=["en", "de", "it", "fr"], dirpath="."):
  vocabulary = []
  for lang in languages:
    # Don't compare english with english
    if lang == "en":
      continue
    # filepath = dirpath + "/" + "top_similarities_europarl_%s_%s_common.csv" % ("en", lang)
    filepath = dirpath + "/" + "top_similarities_europarl_%s_%s.csv" % ("en", lang)
    vocabulary_significances = {}
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()
        for row in reader:
          sim1 = float(row[2] if row[2]!="None" else 0)
          sim2 = float(row[3] if row[3]!="None" else 0)
          significance = np.abs(sim1-sim2)
          for w in (row[0], row[1]):
            vocabulary.append(w.lower())
            vocabulary_significances[w.lower()] = significance
  return vocabulary_significances

def build_training_set(languages=["en", "de", "it", "fr"], dir_path="models"):
    texts = []
    labels = []
    for lang in languages:
      filename = dir_path + "/" + "europarl.%s.en.txt" % lang
      current_texts = load_texts(filename)
      texts.extend(current_texts)
      labels.extend([lang for l in current_texts])
    return texts, labels

def get_support_coefs(clf):
    svm = clf.named_steps['clf']
    vectorizer = clf.named_steps['vect']
    class_labels=svm.classes_
    print svm.coef_.shape, svm.classes_.shape
    feature_names = vectorizer.get_feature_names()
    for i in range(len(class_labels) - 1):
        top10 = np.abs(svm.coef_[i].toarray())[0]
        top_features = {
            feature_names[j] :  np.abs(svm.coef_[i,j]) for j in range(len(top10))}
    return top_features

def feature_significance_correlation(dict1, dict2):
  '''Look at how values if first dictionary correlate with values in second dictionary.
  Will be used for checking whether the SVM and the embeddings method show correlation
  in what they consider important discriminative features.'''
  values1 = []
  values2 = []
  for w in dict1:
    if w not in dict1 or w not in dict2:
      continue
    # filter only values with significance > 0.5 obtained with embeddings method
    if dict2[w] < 0.5:
      continue
    values1.append(dict1[w])
    values2.append(dict2[w])
  from matplotlib import pyplot as plt
  plt.scatter(values1, values2)
  plt.show()
  print "Features correlation:", scipy.stats.pearsonr(values1, values2)

if __name__ == '__main__':
    # languages = ["en", "de", "it", "fr"]
    languages = ["en", "fr"]
    vocabulary_with_significance = load_vocabulary(languages)
    print "Significance of vocabulary words as obtained by embeddings method:", vocabulary_with_significance
    vocabulary = vocabulary_with_significance.keys()
    X, y = build_training_set(languages)

    print "All words:"

    print "Classifier with vocabulary (words from embedding analogies):"
    clf = get_word_classifier(vocabulary=vocabulary)
    train(clf, X, y)
    svm_coefs = get_support_coefs(clf)
    feature_significance_correlation(svm_coefs, vocabulary_with_significance)

    print "Classifier with all words:"
    clf = get_word_classifier()
    train(clf, X, y)
    svm_coefs = get_support_coefs(clf)
    feature_significance_correlation(svm_coefs, vocabulary_with_significance)
    
    print "Classifier with most frequent words", len(vocabulary)
    clf = get_word_classifier(max_features=len(vocabulary))
    train(clf, X, y)

    
    print "Classifier with kbest", len(vocabulary)
    kbest = len(vocabulary)
    clf = get_word_classifier(kbest=kbest)
    train(clf, X, y)
    vectorizer = clf.named_steps['vect']
    feature_names = vectorizer.get_feature_names()
    if 'kbest' in clf.named_steps:
        kbest = clf.named_steps['kbest']
        feature_names = np.asarray(feature_names)[kbest.get_support()]
        print "K best features:", feature_names

    print "Only functional words"

    print "Classifier with all functional words:"
    stopwords_all = nltk.corpus.stopwords.words("english")
    clf = get_word_classifier(vocabulary=stopwords_all)
    train(clf, X, y)

    print "Classifier with vocabulary (only functional):"
    vocabulary_stopwords = filter(lambda w: w in stopwords_all, vocabulary)
    clf = get_word_classifier(vocabulary=vocabulary_stopwords)
    print vocabulary_stopwords
    train(clf, X, y)

    print "Classifier with most frequent words (only functional)", len(vocabulary_stopwords)
    clf = get_word_classifier(max_features=len(vocabulary), vocabulary=stopwords_all)
    train(clf, X, y)

    print "Classifier with kbest function words", len(vocabulary_stopwords)
    kbest = len(vocabulary_stopwords)
    clf = get_word_classifier(kbest=kbest, vocabulary=stopwords_all)
    train(clf, X, y)
    vectorizer = clf.named_steps['vect']
    feature_names = vectorizer.get_feature_names()
    if 'kbest' in clf.named_steps:
        kbest = clf.named_steps['kbest']
        feature_names = np.asarray(feature_names)[kbest.get_support()]
        print "K best features:", feature_names