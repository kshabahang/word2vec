import numpy as np
import sys


variant = sys.argv[1]



f = open("word2vec_{}.txt".format(variant), "r")
vecs = f.readlines()
f.close()

vocab = [vecs[i].split()[0] for i in range(1, len(vecs))]
vecs = np.array([np.array(vecs[i].split()[1:]).astype(float) for i in range(1, len(vecs))])

f = open("vocab_word2vec_{}.txt".format(variant), "w")
f.write("\n".join(vocab))
f.close()

np.save("word2vec_{}".format(variant), vecs)


