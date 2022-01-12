time ./word2vec -train ./TASA2.txt -output word2vec_cbow8.txt -save-vocab vocab_word2vec_cbow8.txt -min-count 0 -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 0 -threads 20 -binary 0 -iter 40
