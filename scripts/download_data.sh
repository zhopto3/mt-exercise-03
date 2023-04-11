#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

#mkdir -p $data/grimm
mkdir -p $data/shake

#mkdir -p $data/grimm/raw
mkdir -p $data/shake/raw

#wget https://www.gutenberg.org/files/52521/52521-0.txt
wget https://www.gutenberg.org/cache/epub/100/pg100.txt
#mv 52521-0.txt $data/grimm/raw/tales.txt
mv pg100.txt $data/shake/raw/allworks.txt

# preprocess slightly

#cat $data/grimm/raw/tales.txt | python $base/scripts/preprocess_raw.py > $data/grimm/raw/tales.cleaned.txt
cat $data/shake/raw/allworks.txt | python $base/scripts/preprocess_raw.py > $data/shake/raw/allworks.cleaned.txt

# tokenize, fix vocabulary upper bound

#cat $data/grimm/raw/tales.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
#    $data/grimm/raw/tales.preprocessed.txt

cat $data/shake/raw/allworks.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/shake/raw/allworks.preprocessed.txt

# split into train, valid and test

#head -n 440 $data/grimm/raw/tales.preprocessed.txt | tail -n 400 > $data/grimm/valid.txt
#head -n 840 $data/grimm/raw/tales.preprocessed.txt | tail -n 400 > $data/grimm/test.txt
#tail -n 3075 $data/grimm/raw/tales.preprocessed.txt | head -n 2955 > $data/grimm/train.txt

head -n 440 $data/shake/raw/allworks.preprocessed.txt | tail -n 400 > $data/shake/valid.txt
head -n 840 $data/shake/raw/allworks.preprocessed.txt | tail -n 400 > $data/shake/test.txt
tail -n 3075 $data/shake/raw/allworks.preprocessed.txt | head -n 2955 > $data/shake/train.txt
