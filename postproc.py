import numpy as np
import sys, os
import json

f = open("../config.json", "r")
config = json.loads(f.read())
f.close()


variant = config['word2vec']["VARIANT"]#sys.argv[1]


mem_path = config["memory_path"] 


corpus = config["corpus"] 

win = config["word2vec"]["WINDOW_SIZE"]

if not os.path.isdir(mem_path + "{}/word2vec".format(corpus)):
    os.system("mkdir {}/{}/word2vec".format(mem_path, corpus))


for m in range(1, config["word2vec"]["N_RUNS"]+1):

    f = open(mem_path + "/{}/word2vec/".format(corpus) +"word2vec_{}WIN{}_RUN{}.txt".format(variant, win, m), "r")
    vecs = f.readlines()
    f.close()
    
    vocab = [vecs[i].split()[0] for i in range(1, len(vecs))]
    vecs = np.array([np.array(vecs[i].split()[1:]).astype(float) for i in range(1, len(vecs))])
    
    f = open( "{}/{}/word2vec/vocab_{}{}_{}.txt".format(mem_path, corpus, variant,config["word2vec"]["WINDOW_SIZE"] , m), "w")
    f.write("\n".join(vocab))
    f.close()
    
    np.save("{}/{}/word2vec/embeddings_{}{}_{}".format(mem_path, corpus, variant, config["word2vec"]["WINDOW_SIZE"], m), vecs)
    
    #clean up
    #os.system("rm word2vec_{}{}.txt".format(variant, m))
    os.system("rm " + mem_path + "/{}/word2vec/".format(corpus) +"word2vec_{}WIN{}_RUN{}.txt".format(variant,     win, m))
    #os.system("rm vocab_word2vec_{}{}.txt".format(variant, m))
    os.system("rm " + mem_path + "/{}/word2vec/".format(corpus) +"vocab_word2vec_{}WIN{}_RUN{}.txt".format(variant,     win, m))





