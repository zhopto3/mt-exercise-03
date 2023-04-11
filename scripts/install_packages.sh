#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

tools=$base/tools
mkdir -p $tools

echo "Make sure this script is executed AFTER you have activated a virtualenv"

pip install numpy torch sacremoses nltk

#Need to install dataframe_image if the visualize_ppy.py script is to be used.
pip install dataframe_image

# install Moses scripts for preprocessing

git clone https://github.com/bricksdont/moses-scripts $tools/moses-scripts

# install pytorch examples

git clone https://github.com/pytorch/examples $tools/pytorch-examples
