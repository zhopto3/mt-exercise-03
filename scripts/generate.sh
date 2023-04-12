
#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

#uncomment below to get the sample data for the baseline model
#(cd $tools/pytorch-examples/word_language_model &&
#    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
#        --data $data/shake \
#        --words 100 \
#        --checkpoint $models/model.pt \
#        --outf $samples/sample1
#)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/shake \
        --words 100 \
        --checkpoint $models/model_0dropout.pt \
        --outf $samples/samp_highestPpy
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/shake \
        --words 100 \
        --checkpoint $models/model_point4dropout.pt \
        --outf $samples/samp_lowestPpy
)
