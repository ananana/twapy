"""Example script illustrating how to use the twapy module to compute a temporal word analogy.

Assuming that there are files called '1987.bin' and '1997.bin' in the 'data/' directory,
this script will load those models and find the 1997 analogue of 'reagan' in 1987 (which should
be 'cilnton').
"""

import gensim

def train_embeddings(text_filepath):
    with open(text_filepath) as f:
        sentences = [line.split() for line in f]
    model = gensim.models.Word2Vec(sentences)
    model.save("/tmp/model")
    return model

def main():
    import twapy
    import sys
    model_dir = './models'
    collection = twapy.ModelCollection(model_dir)
    # spaces = twapy.Alignment('natives_cohesive_marked_embeddings100', 'translations_cohesive_marked_embeddings100', collection=collection)
    spaces = twapy.Alignment('europarl.en.en.100.8.5', 'europarl.fr.en.100.8.5', collection=collection)
    spaces.get_top_similarites(min_freq=1, vocab_file="../cohesive_markers_underscored", ignore_same=False)
    # spaces.get_top_similarites(min_freq=100)

if __name__ == '__main__':
    # model = train_embeddings("models/europarl.it.en.txt")
    # print model.most_similar("nation")
    main()