# Causal Lens for controllable text generation using Multiple Control attributes

## Introduction

The base code of this repo is referred from code of EMNLP 2020 paper "Control, Generate and Augment: A Scalabel Framework for Multi-Attributes Controlled Text Generation". 

## Data Download

Please download the YELP restaurants review data from [here ](https://github.com/shentianxiao/language-style-transfer (edited)).

#### Data Preprocessing

To obtain the person number attributes, please run first

```bash
python PronounLabeling.py
```

## Model

#### Training

to train the model please run 

```bash
python Analysis.py
```

The model trained is saved in the bin folder. The name used is the date and the time the experiment is started

#### Generation

To generate new sentences, please run the below

```bash
python generation.py
```

## Evaluation

```bash
python calc_sent_pronoun.py
```
