CORPUS_PATH=$(jq '.memory_path+.corpus+"/"+.corpus+".txt"|tostring' ../config.json | tr -d '"')
MEM_PATH=$(jq '.memory_path+.corpus + "/word2vec/"|tostring' ../config.json | tr -d '"')
WINDOW_SIZE=$(jq '.word2vec.WINDOW_SIZE' ../config.json)
N_NEGATIVE=$(jq '.word2vec.N_NEGATIVE' ../config.json)
N_THREADS=$(jq '.word2vec.N_THREADS' ../config.json)
N_ITER=$(jq '.word2vec.N_ITER' ../config.json)
CONTEXT_SMOOTHING=$(jq '.word2vec.CONTEXT_SMOOTHING' ../config.json)
N_RUNS=$(jq '.word2vec.N_RUNS' ../config.json)
SAVE_AS_BIN=0
MIN_COUNT=0
for run_i in `seq 1 $N_RUNS`
do
	echo "run $run_i"
	time ./word2vec -train $CORPUS_PATH -output $MEM_PATH'word2vec_cbowWIN'$WINDOW_SIZE'_RUN'$run_i.txt -save-vocab $MEM_PATH'vocab_word2vec_cbowWIN'$WINDOW_SIZE'_RUN'$run_i.txt -min-count $MIN_COUNT -cbow 1 -size 200 -window $WINDOW_SIZE -negative $N_NEGATIVE -hs 0 -sample 0 -threads $N_THREADS -binary $SAVE_AS_BIN -iter $N_ITER -context-smoothing $CONTEXT_SMOOTHING -prop-train 1.0
done
