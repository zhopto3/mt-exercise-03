#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models
mkdir -p $models/logs

num_threads=4

device=""

#BASELINE CONDITIONS USED TO GENERATE "model.pt"
#SECONDS=0
#(cd $tools/pytorch-examples/word_language_model &&
#    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
#        --epochs 40 \
#        --log-interval 100 \
#        --emsize 200 --nhid 200 --dropout 0.5 --tied \
#        --save $models/model.pt
#)

#echo "time taken:"
#echo "$SECONDS seconds"


#Commands for training five models with varying dropout rates:
SECONDS=0
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.0 --tied \
        --save $models/model_0dropout.pt \
        --log-perplexity $models/logs/point0Perplexity.csv
)

echo "time taken:"
echo "$SECONDS seconds"

SECONDS=0
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.2 --tied \
        --save $models/model_point2dropout.pt \
        --log-perplexity $models/logs/point2Perplexity.csv
)

echo "time taken:"
echo "$SECONDS seconds"

SECONDS=0
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.4 --tied \
        --save $models/model_point4dropout.pt \
        --log-perplexity $models/logs/point4Perplexity.csv
)

echo "time taken:"
echo "$SECONDS seconds"

SECONDS=0
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.6 --tied \
        --save $models/model_point6dropout.pt \
        --log-perplexity $models/logs/point6Perplexity.csv
)

echo "time taken:"
echo "$SECONDS seconds"

SECONDS=0
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/shake \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.8 --tied \
        --save $models/model_point8dropout.pt \
        --log-perplexity $models/logs/point8Perplexity.csv
)

echo "time taken:"
echo "$SECONDS seconds"