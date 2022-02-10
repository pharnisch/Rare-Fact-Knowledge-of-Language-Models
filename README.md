# Evaluating Knowledge of Rare Facts in Language Models

This repository contains a framework to analyze the capability of a LM to learn rare fact knowledge.
It is part of the masters thesis for computer science studies of Philipp Lars Harnisch.

## Running the Code

### Installing required packages

```
pip install -r requirements.txt
```

### Downloading and cleaning the pretraining data (approx. 18 GB)

```
python setup.py load-and-clean
```
Data files containing each a maximum of texts of 10.000 Wikipedia articles, and are stored under "/training/data/wikipedia/20200501.en/".

### (Pre)Training

#### Tokenizer Model
```
python setup.py train-tokenizer <tokenizer_name>
```
Possible options for tokenizer_name are ```byte_level_bpe``` (e.g. for RoBERTa) and ```word_piece``` (e.g. for BERT) möglich.

#### Transformer Language Model
```
python train.py BERT
```
Please see ```python train.py --help``` for more options.


### Evaluation

```
python eval.py BERT --max-questions-per-file=-1
```
Please see ```python eval.py --help``` for more options.


## Further Information

### Fact Frequencies

The fact frequencies are required to calculate the metrics with ```python eval.py``` and are already part of the repository.
They were generated with the following commands, using a pool of worker threads in the size of the available cores, subpartioning the counting task into 100-fact-pieces of a file:
```
python setup.py calc-freqs --concept-net [--random-order]
python setup.py calc-freqs --google-re [--random-order]
python setup.py calc-freqs --t-rex [--random-order]
```

For example, ```date_of_birth_frequencies_0.jsonl``` contains the frequencies of the first 100 date_of_birth facts (index 0 to 99).