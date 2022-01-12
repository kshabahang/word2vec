
MEM_PATH=$(cat memory_path.txt)
CORPUS=$(cat corpus.txt)
WINDOW_SIZE=1
N_NEGATIVE=25
N_THREADS=8
N_ITER=10
SAVE_AS_BIN=0
MIN_COUNT=0
time ./word2vec -train $MEM_PATH/$CORPUS/$CORPUS.txt -output word2vec_cbow$WINDOW_SIZE.txt -save-vocab vocab_word2vec_cbow$WINDOW_SIZE.txt -min-count $MIN_COUNT -cbow 1 -size 200 -window $WINDOW_SIZE -negative $N_NEGATIVE -hs 0 -sample 0 -threads $N_THREADS -binary $SAVE_AS_BIN -iter $N_ITER
