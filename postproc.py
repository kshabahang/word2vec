import numpy as np
import sys, os
import json

f = open("../config.json", "r")
config = json.loads(f.read())
f.close()


variant = config['word2vec']["VARIANT"]#sys.argv[1]


mem_path = config["memory_path"] 


corpus = config["corpus"] 


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





