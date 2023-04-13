# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/moritz-steiner/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

install_packages.sh has been updated to install the necessary packes used to log model perplexities during training.

Download and preprocess data. This script will collect the complete works of Shakespeare from the internet and create a correctly
preprocessed test, train, and validation set from them:

    ./scripts/download_data.sh

Prior to training the model, you must first override the word language model "main.py" file with the version that can log model perplexities during training. First find the following script:

    ./scripts/main.py

Now move the above script to the following path, replacing the main.py script that is already in the folder:


    /tools/pytorch-examples/word_language_model/main.py

Running the following script will now train and evaluate five language models with verying levels of dropout, using the training, validation, and test data created from the works of Shakespeare. It will also create logs of each models perplexity at each epoch, as calculated for the test, validation, and data set. The perplexity after training will be calculated using the test set and logged as well:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from the trained model with 0.0 dropout (which we found to have the highest test set perplexity) and the trained model with 0.4 dropout (which we found resulted in the lowest test set perplexity):

    ./scripts/generate.sh

The samples generated from the model trained on default settings will be saved in ./samples

