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
python training/data/load_and_clean_data.py
```
Data files containing each a maximum of texts of 10.000 Wikipedia articles, and are stored under "/training/data/wikipedia/20200501.en/".

### Training

```
python train.py BERT
```
Please see ```python train.py --help``` for more options.


### Evaluating

```
python eval.py BERT -k 10
```
Please see ```python eval.py --help``` for more options.


## Further Information

The fact frequencies are required to calculate the metrics with ```python eval.py``` and are already part of the repository.
They were generated with ```python setup.py calc-freqs --concept-net --google-re --t-rex``` of the general command
```
python setup.py calc-freqs [--concept-net] [--google-re] [--t-rex] [--max-files=int] [--max-questions-per-file=int] [--verbose]
```