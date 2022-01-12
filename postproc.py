import numpy as np
import sys, os


variant = sys.argv[1]

f = open("memory_path.txt", "r")
mem_path = f.read().split('\n')[0]
f.close()
f = open("corpus.txt", "r")
corpus = f.read().split('\n')[0]
f.close()

if not os.path.isdir(mem_path + "{}/word2vec".format(corpus)):
    os.system("mkdir {}/{}/word2vec".format(mem_path, corpus))




f = open("word2vec_{}.txt".format(variant), "r")
vecs = f.readlines()
f.close()

vocab = [vecs[i].split()[0] for i in range(1, len(vecs))]
vecs = np.array([np.array(vecs[i].split()[1:]).astype(float) for i in range(1, len(vecs))])

f = open( "{}/{}/word2vec/vocab_{}.txt".format(mem_path, corpus, variant), "w")
f.write("\n".join(vocab))
f.close()

np.save("{}/{}/word2vec/embeddings_{}".format(mem_path, corpus, variant), vecs)

#clean up
os.system("rm word2vec_{}.txt".format(variant))
os.system("rm vocab_word2vec_{}.txt".format(variant))





