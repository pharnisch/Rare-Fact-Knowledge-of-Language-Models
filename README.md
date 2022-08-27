# Evaluating the Sample-Efficiency of Language Models for Learning Factual Knowledge (Master Thesis Project)

This repository contains a framework to analyze the capability of a LM to learn rare fact knowledge. In this context, the term fact refers to commonsense or factual knowledge in the form of a subject-relation-object triple. The code includes pipelines to
- i) download and clean the pre-training corpus (English Wikipedia),
- ii) train Hugging Face transformer language models, 
- iii) estimate frequencies of facts from the knowledge bases ConceptNet, Google-RE and T-REx, 
- iv) calculate evaluation results like correlation coefficients between estimated fact frequencies and the respective model's precision at one score.
It is part of the masters thesis for computer science studies of Philipp Lars Harnisch.

## Running the Code

### Installing required packages

```
pip install -r requirements.txt
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Downloading and cleaning the pretraining data (approx. 18 GB)

```
python setup.py load-and-clean
```
Data files containing each a maximum of texts of 10.000 Wikipedia articles, and are stored under "/training/data/wikipedia/20200501.en/". This makes lazy-loading of the data possible within processing loops like the pre-training or statistics calculations.

### (Pre)Training

#### Tokenizer Model
```
python setup.py train-tokenizer <tokenizer_name>
```
e.g.:
```
python setup.py train-tokenizer word_piece
```
Possible options for tokenizer_name are ```byte_level_bpe``` (e.g. for RoBERTa) and ```word_piece``` (e.g. for BERT).

#### Transformer Language Model
```
python train.py <model_name> [--fresh-start] [--no-eval] [--seed=1337] [--epochs=10] [--learning-rate=0.001] [--cuda-index=0] [--batch_size=16] [--num-hidden-layers=12] [--training-data-rate=1] [--accumulated_batches=256]
```
e.g.:
```
python train.py CorBert
```
A checkpoint is saved for every epoch in directory ```/models```. The command searches for identical checkpoints and will continue the latest model state if there is any. Possible options for model_name are ```CorBert```, ```CorRoberta```, ```CorElectra``` and ```CorDistilBert```. Please see ```python train.py --help``` for more options. 

### Evaluation

```
python eval.py <relation_file> [--seed=1337] [--k=1] [--max-questions-per-file=100] [--by-example] [--min-quantile=0] [--max-quantile=1]
```
e.g.:
```
python eval.py BERT-6-1-16-0.0001-20-2.503282-checkpoint.pth --max-questions-per-file=-1
```
This command creates a json-file in directory ```/metrics/standard``` containing all single fact data points of the evaluated relation type, and some aggregations like the Pearson and Spearman correlation coefficient. ```--k=1``` refers to the implementation of the precision at one metric which can be changed to other upper rank boundaries to count successful predictions. ```--by-example``` enables a probing enhancement method of prepending 10 randomly selected examples of the same relation type. With ```--min-quantile``` and  ```--max-quantile``` it can be controlled how rare or common the fact examples should be, the quantile refers the the dimension of estimated fact frequencies.
Please see ```python eval.py --help``` for more options.

#### Visualization


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
