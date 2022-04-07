import os, json
import numpy as np
import pickle

def prepareFAs(fas, I):
    cue_resps = []
    K = 1
    for cue in fas.keys():
        resps, ps = zip(*fas[cue])
        idxs = np.argsort(ps)[::-1]
        idx = np.argmax(ps)
        first_p = ps[idx]
        first_resps = []
        if len(idxs) >= K:
            for j in range(len(resps)):
                first_resps.append(resps[idxs[j]])

            first_resp = resps[idx]
            #fa_vocab.append(first_resp)

            cue_resps.append((first_p, cue, first_resp, first_resps))



    cue_resps = sorted(cue_resps)[::-1]

    cue_resps_try = []
    fa_vocab = []
    k = 0
    i = 0
    while(k < 1000000 and i < len(cue_resps)):
        (p, cue, resp, _) = cue_resps[i]
        if cue in I and resp in I:
            cue_resps_try.append((p, cue, resp))
            fa_vocab.append(cue)
            fa_vocab.append(resp)
            k += 1
            i += 1
        else:
            i += 1
    fa_vocab = list(set(fa_vocab))

    return sorted(fa_vocab), cue_resps_try, cue_resps

def vecs2cos(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    return np.diag(1/norms).dot(vectors.dot(vectors.T)).dot(np.diag(1/norms))


f = open("sweep.json", "r")
sweep = json.loads(f.read())
f.close()


f = open("local_config.json", "r")
local_config = json.loads(f.read())
f.close()



FA_FILE = local_config["FA_FILE"]

f = open(FA_FILE, "rb")
fas = pickle.load(f)
f.close()

fas_sub = {cue:fas[cue] for cue in list(fas.keys())[:10]}


N_RUNS = local_config["N_RUNS"]
CORPUS_PATH = local_config["CORPUS_PATH"]
CORPUS_FILE = local_config["CORPUS_FILE"]
MEM_PATH = local_config["MEM_PATH"]
OUT_FILE = local_config["OUT_FILE"]
OUT_FILE_VOCAB = local_config["OUT_FILE_VOCAB"]
N_THREADS = local_config["N_THREADS"]
N_ITER = local_config["N_ITER"]
DIMENSIONALITY = local_config["DIMENSIONALITY"]


arg_lbls = ["-train", 
            "-output", 
            "-save-vocab", 
            "-min-count", 
            "-cbow", 
            "-size", 
            "-window", 
            "-negative",
            "-hs",
            "-sample",
            "-threads",
            "-binary",
            "-iter",
            "-context-smoothing"]

print("window-size alpha k med_ranks sterr(med_ranks) pfirst sterr(pfirst)")

for window_size in sweep["WINDOW_SIZE"]:
    for alpha in sweep["alpha"]:
        for k in sweep["k"]:

            arg_vals = [CORPUS_PATH + CORPUS_FILE,
                        MEM_PATH + OUT_FILE,
                        MEM_PATH + OUT_FILE_VOCAB,
                        0,
                        1,
                        DIMENSIONALITY,
                        int((window_size - 1)/2),
                        k,
                        0,
                        0,
                        N_THREADS,
                        0,
                        N_ITER,
                        alpha]
            
            args = ' '.join(["{} {}".format(arg_lbls[i], arg_vals[i]) for i in range(len(arg_vals))])
            
            res_by_run = {"med_ranks":[], "pfirst":[]}
            for i in range(N_RUNS):
            
                os.system("./word2vec " + args + " > train.out")
                
                f = open(MEM_PATH + OUT_FILE, "r")
                vecs = f.readlines()
                f.close()
                
                vocab = [vecs[i].split()[0] for i in range(1, len(vecs))]
                vecs = np.array([np.array(vecs[i].split()[1:]).astype(float) for i in range(1, len(vecs))])
            
                I = {vocab[i]:i for i in range(len(vocab))}
            
                fa_vocab, cue_resps_try, cue_resps = prepareFAs(fas, I)
                
                idxs = np.array([I[fa_vocab[i]] for i in range(len(fa_vocab))])
                
                I = {fa_vocab[i]:i for i in range(len(fa_vocab))}
             
            
                wXw = vecs2cos(vecs[idxs]) - np.eye(len(fa_vocab))
            
                ranks_w2v = []
                for i in range(len(cue_resps_try)):
                    (p, cue, resp1) = cue_resps_try[i]
                    r_w2v = list(np.argsort(wXw[I[cue]])[::-1]).index(I[resp1])
                    ranks_w2v.append(r_w2v)
                
                ranks_w2v = np.array(ranks_w2v)
            
                res_by_run["med_ranks"].append(np.median(ranks_w2v))
                res_by_run["pfirst"].append(100*sum(ranks_w2v == 0)/len(ranks_w2v))
            
            print(window_size, alpha, k, round(np.mean(res_by_run["med_ranks"]), 3),
                    round(np.std(res_by_run["med_ranks"])/np.sqrt(len(res_by_run["med_ranks"])), 3), 
                    round(np.mean(res_by_run["pfirst"]),3),
                    round(np.std(res_by_run["pfirst"])/np.sqrt(len(res_by_run["pfirst"])),3))
