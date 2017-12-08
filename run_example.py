"""Example script illustrating how to use the twapy module to compute a temporal word analogy.

Assuming that there are files called '1987.bin' and '1997.bin' in the 'data/' directory,
this script will load those models and find the 1997 analogue of 'reagan' in 1987 (which should
be 'cilnton').
"""

import twapy
import sys

model_dir = './models'
collection = twapy.ModelCollection(model_dir)
while True:
    word = sys.stdin.readline().strip()
    analogy = twapy.Analogy(word, 'natives_embeddings100', 'translations_embeddings100', collection=collection)
    print(analogy)
# This prints the following:
# 1987 : reagan :: 1997 : clinton
