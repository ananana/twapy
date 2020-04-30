"""Example script illustrating how to use the twapy module to compute a temporal word analogy.

Assuming that there are files called '1987.bin' and '1997.bin' in the 'data/' directory,
this script will load those models and find the 1997 analogue of 'reagan' in 1987 (which should
be 'cilnton').
"""

import gensim
from convert import convert_gensim
from nltk import word_tokenize

def train_embeddings(text_filepath):
    with open("./models/" + text_filepath) as f:
        sentences = [[w.lower() for w in word_tokenize(line)] for line in f]
    model = gensim.models.Word2Vec(sentences)
    text_filepath_noext = ".".join(text_filepath.split(".")[:-1])
    model.wv.save_word2vec_format("./models/" + text_filepath_noext + ".wv", binary=False) # alignment.py expects it to have an extension
    return model

def finetune_embeddings(text_filepath):
    google_wv = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/ana/code/research/tools/embeddings/GoogleNews-vectors-negative300.bin', 
    binary=True)
    model = gensim.models.Word2Vec(size=300, min_count=5, iter=10)
    with open("./models/" + text_filepath) as f:
        sentences = [[w.lower() for w in word_tokenize(line)] for line in f]
    model.build_vocab(sentences)
    training_examples_count = model.corpus_count
    # below line will make it 1, so saving it before
    model.build_vocab([list(google_wv.vocab.keys())], update=True)
    model.intersect_word2vec_format(
        "/home/ana/code/research/tools/embeddings/GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
    model.train(sentences,total_examples=training_examples_count, epochs=model.iter)


    # model.save("word2vec_model2")
    #model1 = Word2Vec.load("word2vec_model")
    # model.wv.save("finetuned_embeddings_" + text_filepath + ".bin")
    # model.wv.save_word2vec_format("finetuned_%s_word2vec_model_vectors2.bin" % text_filepath)
    text_filepath_noext = ".".join(text_filepath.split(".")[:-1])
    model.wv.save_word2vec_format("./models/" + text_filepath_noext + ".wv", binary=False) # alignment.py expects it to have an extension

    #word_vectors1 = KeyedVectors.load("word2vec_model_vectors")

def main():
    import twapy
    import sys
    model_dir = './models'
    collection = twapy.ModelCollection(model_dir)
    # spaces = twapy.Alignment('natives_cohesive_marked_embeddings100', 'translations_cohesive_marked_embeddings100', collection=collection)
    # spaces = twapy.Alignment('europarl_and_hansard_fr.en.100', 'europarl_and_hansard_en.en.100', collection=collection)
    # spaces = twapy.Alignment('europarl_and_hansard_fr.en', 'europarl_and_hansard_en.en', collection=collection)
    spaces = twapy.Alignment('nonnatives_fr', 'natives_fr', collection=collection)
    # spaces = twapy.Alignment('europarl.en.en.100.8.5', 'europarl.fr.en.100.8.5', collection=collection)
    # spaces.get_top_similarites(min_freq=1, vocab_file="../cohesive_markers_underscored", ignore_same=False)
    spaces.get_top_similarites(min_freq=50)

if __name__ == '__main__':
    # model = finetune_embeddings("natives_fr.txt")
    # model = finetune_embeddings("natives_fr.txt")
    # model = train_embeddings("natives_fr.txt")
    # model = train_embeddings("nonnatives_fr.txt")
    # print model.most_similar("nation")
    main()