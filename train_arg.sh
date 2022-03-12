CORPUS_PATH=$(jq '.memory_path+.corpus+"/"+.corpus+".txt"|tostring' ../config.json | tr -d '"')
WINDOW_SIZE=$1 #$(jq '.word2vec.WINDOW_SIZE' ../config.json)
N_NEGATIVE=$(jq '.word2vec.N_NEGATIVE' ../config.json)
N_THREADS=$(jq '.word2vec.N_THREADS' ../config.json)
N_ITER=$(jq '.word2vec.N_ITER' ../config.json)
N_RUNS=2 #$(jq '.word2vec.N_RUNS' ../config.json)
SAVE_AS_BIN=0
MIN_COUNT=0
for run_i in `seq 1 $N_RUNS`
do
	echo "run $run_i"
	time ./word2vec -train $CORPUS_PATH -output word2vec_cbow$WINDOW_SIZErun$run_i.txt -save-vocab vocab_word2vec_cbow$WINDOW_SIZErun$run_i.txt -min-count $MIN_COUNT -cbow 1 -size 200 -window $WINDOW_SIZE -negative $N_NEGATIVE -hs 0 -sample 0 -threads $N_THREADS -binary $SAVE_AS_BIN -iter $N_ITER
done
